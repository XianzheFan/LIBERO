"""
Train a DINOv2-based Value Expert (following DreamDojo's external value model).

Architecture:
  - Input: a 4-frame video clip (DreamDojo output)
  - Frozen DINOv2 ViT-B/14 extracts per-frame CLS features for all 4 images
  - A learnable Transformer encoder with global attention fuses the temporal features
  - A small MLP head outputs a **scalar value**: normalized remaining time steps to subtask boundary

Supervision:
  - For each clip sampled from a demonstration, the label is
        remaining_steps_to_subtask_boundary / max_subtask_interval
    where subtask boundaries are defined by language annotation switches in the dataset.
  - Lower value → closer to completing the current subtask.

Training data format (directory of .npz files, each containing):
  - "video_clip":        uint8 (4, H, W, 3)    -- 4 consecutive frames
  - "value":             float32 scalar         -- normalized remaining steps (0=done, 1=far)

Usage:
  python train_dinov2_value_expert.py \
      --data_dir data/value_expert_dataset \
      --output_dir checkpoints/dinov2_value_expert \
      --epochs 50 --batch_size 64 --lr 1e-4
"""

import argparse
import glob
import logging
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class DINOv2ValueExpert(nn.Module):
    """
    DINOv2-based value model for scoring short video clips.

    Inputs
    ------
    video_clip : (B, num_clip_frames, 3, H, W)   – e.g. 4 consecutive frames from DreamDojo

    Output
    ------
    value : (B,) – normalized remaining steps to subtask completion.  Lower = better.
    """

    def __init__(
        self,
        num_clip_frames: int = 4,
        dinov2_model: str = "dinov2_vitb14",
        attn_heads: int = 8,
        attn_layers: int = 2,
        hidden_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_clip_frames = num_clip_frames
        self.num_tokens = num_clip_frames  # 4

        # ---- Frozen DINOv2 backbone ----
        self.backbone = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.feature_dim = self.backbone.embed_dim  # 768 for ViT-B/14
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # DINOv2 expects ImageNet normalization
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # ---- Learnable positional embedding for clip frames ----
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, self.feature_dim) * 0.02)

        # ---- Global attention (Transformer encoder) ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=attn_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=attn_layers)
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )


    def _normalize_and_resize(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs : (N, 3, H, W) float32 in [0, 1]
        Returns: (N, 3, 224, 224) normalized for DINOv2
        """
        imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
        imgs = (imgs - self.img_mean) / self.img_std
        return imgs

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, 3, H, W) float32 in [0, 1]
        Returns: (B, T, feature_dim)
        """
        B, T, C, H, W = frames.shape
        flat = frames.reshape(B * T, C, H, W)
        flat = self._normalize_and_resize(flat)
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(flat)  # (B*T, feature_dim)
        else:
            feats = self.backbone(flat)
        return feats.reshape(B, T, self.feature_dim)

    def forward(
        self,
        video_clip: torch.Tensor,
    ) -> torch.Tensor:
        """
        video_clip : (B, num_clip_frames, 3, H, W) float32 [0,1]
        Returns: (B,) value scores
        """
        clip_feats = self._encode_frames(video_clip)   # (B, 4, D)

        tokens = clip_feats + self.pos_embed  # (B, 4, D)

        # Global attention
        tokens = self.temporal_attn(tokens)  # (B, 4, D)

        # Mean pool over all tokens → value
        pooled = tokens.mean(dim=1)  # (B, D)
        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return value

    def score_video(
        self,
        video_frames: torch.Tensor,
        window_size: int = 4,
        stride: int = 1,
    ) -> torch.Tensor:
        """
        Score an entire generated video using stride-1 sliding windows.

        video_frames : (B, L, 3, H, W)   – L frames of generated / real video

        Returns: (B, num_windows) per-window values
        """
        B, L, C, H, W = video_frames.shape
        num_windows = (L - window_size) // stride + 1
        all_values = []

        for i in range(0, num_windows * stride, stride):
            clip = video_frames[:, i : i + window_size]  # (B, 4, 3, H, W)
            v = self.forward(clip)                        # (B,)
            all_values.append(v)

        return torch.stack(all_values, dim=1)  # (B, num_windows)


class ValueExpertDataset(Dataset):
    """Dataset of (4-frame clip, value label) from .npz files."""

    def __init__(self, data_dir: str, num_clip_frames: int = 4):
        self.data_dir = pathlib.Path(data_dir)
        self.files = sorted(glob.glob(str(self.data_dir / "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        self.num_clip_frames = num_clip_frames
        logging.info(f"Found {len(self.files)} samples in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        # Video clip: (4, H, W, 3) uint8 → (4, 3, H, W) float [0,1]
        clip = torch.from_numpy(data["video_clip"]).permute(0, 3, 1, 2).float() / 255.0
        if clip.shape[0] < self.num_clip_frames:
            pad = clip[-1:].expand(self.num_clip_frames - clip.shape[0], -1, -1, -1)
            clip = torch.cat([clip, pad], dim=0)
        clip = clip[: self.num_clip_frames]

        value = torch.tensor(float(data["value"]), dtype=torch.float32)

        return clip, value


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = DINOv2ValueExpert(
        num_clip_frames=args.num_clip_frames,
        dinov2_model=args.dinov2_model,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        hidden_dim=args.hidden_dim,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    train_dataset = ValueExpertDataset(args.data_dir, num_clip_frames=args.num_clip_frames)
    val_dataset = ValueExpertDataset(args.val_dir, num_clip_frames=args.num_clip_frames) if args.val_dir else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=True)
        if val_dataset else None
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # ---- Train ----
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()

        total_loss, num_batches = 0.0, 0
        for clip, values in train_loader:
            clip, values = clip.to(device), values.to(device)
            pred = model(clip)
            loss = F.mse_loss(pred, values)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train = total_loss / max(num_batches, 1)
        scheduler.step()

        # ---- Val ----
        val_str = "N/A"
        if val_loader is not None:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for clip, values in val_loader:
                    clip, values = clip.to(device), values.to(device)
                    pred = model(clip)
                    vl += F.mse_loss(pred, values).item()
                    vn += 1
            avg_val = vl / max(vn, 1)
            val_str = f"{avg_val:.6f}"
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                logging.info(f"  -> Saved best model (val_loss={avg_val:.6f})")

        logging.info(
            f"Epoch {epoch+1}/{args.epochs}  train={avg_train:.6f}  val={val_str}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), output_dir / "model_final.pt")
    logging.info(f"Training complete. Models saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train DINOv2 Value Expert (DreamDojo-style)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/dinov2_value_expert")
    # Model
    parser.add_argument("--num_clip_frames", type=int, default=4)
    parser.add_argument("--dinov2_model", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--attn_heads", type=int, default=8)
    parser.add_argument("--attn_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(args)


if __name__ == "__main__":
    main()
