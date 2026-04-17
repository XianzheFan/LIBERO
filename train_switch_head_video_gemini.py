"""
Video-input switch head: uses multi-view video clips (not single frames) to predict
whether the robot needs intervention, matching the temporal context that Gemini uses.

Data collection is handled by:
  - train_switch_head_gemini.py collect           (LIBERO simulation)
  - agilex_collect_switch_labels_gemini.py        (Agilex real robot)
Both save single-frame images AND video clips in the same .npz, so the same data
works for both the image-input model (train_switch_head_gemini.py train) and this
video-input model.

Model architecture:
  Per camera:
    DINOv2 (frozen) encodes each frame → (B, T, 768)
    → learnable positional embedding + Transformer encoder (temporal attention)
    → mean pool over time → (B, 768)
  Concat all camera features + robot state → MLP classifier → switch logit

Usage
-----
  # Collect data (same command for both image and video training)
  python train_switch_head_gemini.py collect \
      --host 10.0.0.1 --port 8000 \
      --output_dir data/switch_labels \
      --clip_len 20

  # Train video-input switch head
  python train_switch_head_video_gemini.py train \
      --data_dir data/switch_labels \
      --output_dir checkpoints/switch_head_video \
      --epochs 30
"""

import argparse
import glob
import logging
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
#  Model
# ============================================================================

class DINOv2VideoSwitchHead(nn.Module):
    """
    DINOv2 + temporal attention switch head for multi-view video clips.

    Per camera:
      1. Frozen DINOv2 encodes each frame → (B, T, D) CLS features
      2. Learnable positional embedding + Transformer encoder fuses time
      3. Mean-pool over time → (B, D)
    Then concatenate all camera features + state → classifier → switch logit.
    """

    def __init__(
        self,
        num_cameras: int = 2,
        clip_len: int = 20,
        dinov2_model: str = "dinov2_vitb14",
        attn_heads: int = 8,
        attn_layers: int = 2,
        temporal_dim: int = 512,
        classifier_dim: int = 256,
        state_dim: int = 7,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.clip_len = clip_len

        # ---- Frozen DINOv2 backbone ----
        self.backbone = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.feature_dim = self.backbone.embed_dim  # 768 for ViT-B/14
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # ---- Temporal attention (shared across cameras) ----
        self.pos_embed = nn.Parameter(
            torch.randn(1, clip_len, self.feature_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=attn_heads,
            dim_feedforward=temporal_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(
            encoder_layer, num_layers=attn_layers
        )

        # ---- Classifier: concat all camera features + state ----
        input_dim = num_cameras * self.feature_dim + state_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, classifier_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_dim, classifier_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_dim, 1),
        )

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, 3, H, W) float [0,1]
        Returns: (B, T, D) per-frame CLS features
        """
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        flat = F.interpolate(flat, size=(224, 224), mode="bilinear", align_corners=False)
        flat = (flat - self.img_mean) / self.img_std
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(flat)  # (B*T, D)
        else:
            feats = self.backbone(flat)
        return feats.reshape(B, T, self.feature_dim)

    def _temporal_pool(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """
        frame_feats: (B, T, D)
        Returns: (B, D) — temporally-attended and pooled representation
        """
        T = frame_feats.shape[1]
        tokens = frame_feats + self.pos_embed[:, :T, :]
        tokens = self.temporal_attn(tokens)  # (B, T, D)
        return tokens.mean(dim=1)            # (B, D)

    def forward(
        self,
        video_clips: list[torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            video_clips: list of (B, T, 3, H, W) float [0,1], one per camera
            state:       (B, state_dim)
        Returns:
            logit: (B,) — raw logit (sigmoid → switch probability)
        """
        cam_feats = []
        for clip in video_clips:
            frame_feats = self._encode_frames(clip)     # (B, T, D)
            pooled = self._temporal_pool(frame_feats)    # (B, D)
            cam_feats.append(pooled)

        combined = torch.cat(cam_feats + [state], dim=-1)
        return self.classifier(combined).squeeze(-1)

    def predict_switch_prob(
        self, video_clips: list[torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(video_clips, state))


# ============================================================================
#  Dataset
# ============================================================================

class SwitchVideoDataset(Dataset):
    """
    Loads .npz files with video clips.

    Auto-detects format:
      - LIBERO: "image_clip", "wrist_clip"              → 2 cameras
      - Agilex: "top_clip", "right_clip", "left_clip"   → 3 cameras
    """

    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No sample_*.npz files in {data_dir}")
        logging.info(f"Found {len(self.files)} samples in {data_dir}")

        first = np.load(self.files[0], allow_pickle=True)
        if "top_clip" in first:
            self.format = "agilex"
            self.num_cameras = 3
            self.clip_keys = ["top_clip", "right_clip", "left_clip"]
        else:
            self.format = "libero"
            self.num_cameras = 2
            self.clip_keys = ["image_clip", "wrist_clip"]

        self.clip_len = int(first.get("clip_len", first[self.clip_keys[0]].shape[0]))
        self.state_dim = first["state"].shape[0]
        logging.info(
            f"  Format: {self.format}, {self.num_cameras} cameras, "
            f"clip_len={self.clip_len}, state_dim={self.state_dim}"
        )

        labels = [float(np.load(f)["switch_label"]) for f in self.files]
        n_pos = sum(1 for l in labels if l > 0.5)
        n_neg = len(labels) - n_pos
        logging.info(f"  Class balance: {n_pos} rescue / {n_neg} normal")
        self._labels = labels

    def __len__(self):
        return len(self.files)

    @property
    def pos_weight(self) -> float:
        n_pos = sum(1 for l in self._labels if l > 0.5)
        n_neg = len(self._labels) - n_pos
        return (n_neg / n_pos) if n_pos > 0 else 1.0

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        # (T, H, W, 3) uint8 → (T, 3, H, W) float [0,1]
        clips = []
        for key in self.clip_keys:
            clip = torch.from_numpy(data[key]).permute(0, 3, 1, 2).float() / 255.0
            clips.append(clip)

        state = torch.from_numpy(data["state"].astype(np.float32))
        label = torch.tensor(float(data["switch_label"]), dtype=torch.float32)

        return clips, state, label


def _collate_video(batch):
    """Custom collate: stack per-camera clips into batched tensors."""
    clips_list, states, labels = zip(*batch)
    num_cameras = len(clips_list[0])
    batched_clips = [
        torch.stack([sample[c] for sample in clips_list])
        for c in range(num_cameras)
    ]
    return batched_clips, torch.stack(states), torch.stack(labels)


# ============================================================================
#  Training
# ============================================================================

def train(args):
    """Train video-input switch head."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_dataset = SwitchVideoDataset(args.data_dir)
    val_dataset = SwitchVideoDataset(args.val_dir) if args.val_dir else None

    model = DINOv2VideoSwitchHead(
        num_cameras=train_dataset.num_cameras,
        clip_len=train_dataset.clip_len,
        dinov2_model=args.dinov2_model,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        temporal_dim=args.temporal_dim,
        classifier_dim=args.classifier_dim,
        state_dim=train_dataset.state_dim,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_video,
    )
    val_loader = (
        DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=_collate_video,
        )
        if val_dataset else None
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    pw = torch.tensor([train_dataset.pos_weight], device=device)
    logging.info(f"BCE pos_weight: {pw.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -float("inf")

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()

        total_loss, num_batches = 0.0, 0
        train_correct, train_total = 0, 0

        for clips, state, label in train_loader:
            clips = [c.to(device) for c in clips]
            state, label = state.to(device), label.to(device)

            logit = model(clips, state)
            loss = criterion(logit, label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pred = (torch.sigmoid(logit) > 0.5).float()
            train_correct += (pred == label).sum().item()
            train_total += label.numel()

        avg_train_loss = total_loss / max(num_batches, 1)
        train_acc = train_correct / max(train_total, 1)
        scheduler.step()

        # ---- Val ----
        val_str = "N/A"
        if val_loader is not None:
            model.eval()
            vl, vn = 0.0, 0
            val_tp, val_fp, val_fn = 0, 0, 0

            with torch.no_grad():
                for clips, state, label in val_loader:
                    clips = [c.to(device) for c in clips]
                    state, label = state.to(device), label.to(device)

                    logit = model(clips, state)
                    vl += criterion(logit, label).item()
                    vn += 1

                    pred = (torch.sigmoid(logit) > 0.5).float()
                    val_tp += ((pred == 1) & (label == 1)).sum().item()
                    val_fp += ((pred == 1) & (label == 0)).sum().item()
                    val_fn += ((pred == 0) & (label == 1)).sum().item()

            avg_val_loss = vl / max(vn, 1)
            val_precision = val_tp / max(val_tp + val_fp, 1)
            val_recall = val_tp / max(val_tp + val_fn, 1)
            val_f1 = 2 * val_precision * val_recall / max(val_precision + val_recall, 1e-8)

            val_str = (
                f"loss={avg_val_loss:.4f} P={val_precision:.3f} "
                f"R={val_recall:.3f} F1={val_f1:.3f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logging.info(f"  -> Saved best model (F1={val_f1:.4f})")

        logging.info(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val=[{val_str}]  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), output_dir / "model_final.pt")
    logging.info(f"Training complete. Models saved to {output_dir}")


# ============================================================================
#  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train video-input DINOv2 switch head"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    p_t = subparsers.add_parser("train",
                                help="Train video-input switch head on collected data")
    p_t.add_argument("--data_dir", type=str, required=True)
    p_t.add_argument("--val_dir", type=str, default=None)
    p_t.add_argument("--output_dir", type=str, default="checkpoints/switch_head_video")
    p_t.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                      choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    p_t.add_argument("--attn_heads", type=int, default=8)
    p_t.add_argument("--attn_layers", type=int, default=2)
    p_t.add_argument("--temporal_dim", type=int, default=512)
    p_t.add_argument("--classifier_dim", type=int, default=256)
    p_t.add_argument("--freeze_backbone", action="store_true", default=True)
    p_t.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    p_t.add_argument("--batch_size", type=int, default=16)
    p_t.add_argument("--lr", type=float, default=1e-4)
    p_t.add_argument("--weight_decay", type=float, default=1e-4)
    p_t.add_argument("--epochs", type=int, default=30)
    p_t.add_argument("--num_workers", type=int, default=4)
    p_t.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
