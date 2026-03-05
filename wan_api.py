import requests
import concurrent.futures
import time

def request_generation(port: int, image_path: str, prompt: str, save_path: str):
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "image_path": image_path,
        "prompt": prompt,
        "save_path": save_path,
        "sampling_steps": 30  # Adjust to increase generation speed
    }
    
    print(f"[Port {port}] Submitting task: {image_path}...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        end_time = time.time()
        print(f"[Port {port}] Task completed! Time elapsed: {end_time - start_time:.2f}s. Saved to: {save_path}")
        return True
    except Exception as e:
        print(f"[Port {port}] Task failed: {e}")
        return False

if __name__ == "__main__":
    shared_prompt = """Please examine the robotic arm workspace in the input image. The drawn trajectory represents the future motion of the gripper's center: the green dot is the starting point, and the red dot is the destination. 
Please imagine a video showing the complete process of the upper part of the robotic arm (including the gripper) moving from the start to the end (keeping the drawn trajectory stationary). Follow these physical rules: 
1. In the final state, the center of the end-effector (gripper) must be perfectly aligned with the red dot. The base of the robotic arm must remain stationary. 
2. Only the upper joints and the arm of the robotic arm move to perform tasks, while the base remains stationary. 
3. Object Interaction Logic: Observe the relationship between the green dot (start position) and any objects in the scene. 
- IF the green dot is currently positioned near an object in a way that implies a grasp or contact, or if the gripper is already holding it: Move that specific object along the trajectory so that it is still being held or manipulated by the gripper at the red dot. 
- IF the green dot and the predicted path are in free space, not touching any objects: Execute the movement as pure free-space motion. No objects should be moved. 
- Do not change the open/closed state of the gripper unless it is necessary for physical plausibility at the destination (e.g., placing an object). 
4. Environment Stability: Keep all other background elements, lighting, camera angle, and non-interacted objects exactly identical to the original scene."""
    
    # Construct task list and distribute to different ports (mapping to different GPUs)
    tasks = [
        {"port": 8010, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_0.png", "save": "./output_0.mp4"},
        {"port": 8011, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_1.png", "save": "./output_1.mp4"},
        {"port": 8012, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_2.png", "save": "./output_2.mp4"},
        {"port": 8013, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_3.png", "save": "./output_3.mp4"},
        {"port": 8014, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_4.png", "save": "./output_4.mp4"},
    ]
    
    print("Initializing parallel Wan2.2 video generation...")
    
    # Use a thread pool to send requests concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = []
        for task in tasks:
            futures.append(
                executor.submit(
                    request_generation, 
                    task["port"], 
                    task["img"], 
                    shared_prompt, 
                    task["save"]
                )
            )
        
        concurrent.futures.wait(futures)
        
    print("All videos generated successfully. Continuing OpenPI workflow!")