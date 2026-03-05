import requests
import concurrent.futures
import time
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

class FrameEvaluation(BaseModel):
    best_index: int

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

def evaluate_videos_with_gemini(current_video_path: str, candidate_paths: list, language_command: str):
    load_dotenv()
    
    client = genai.Client(http_options={'api_version': 'v1alpha'})
    
    print("Uploading videos to Google GenAI...")
    
    current_video_file = client.files.upload(file=current_video_path)
    
    candidate_files = []
    for path in candidate_paths:
        candidate_files.append(client.files.upload(file=path))

    def wait_for_files_active(*files):
        print("Waiting for video processing...")
        for f in files:
            file_info = client.files.get(name=f.name)
            while file_info.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(2)
                file_info = client.files.get(name=f.name)
            if file_info.state.name == "FAILED":
                raise ValueError(f"Video processing failed for file: {f.name}")
        print("\nAll videos are ready!")

    wait_for_files_active(current_video_file, *candidate_files)

    prompt_instruction = f"""
You are an evaluation model in a robotic control system. 
Based on the [Current Video] and the [Language Command], your task is to select the most accurate video from the provided [Candidate Next Videos] that best follows the intended trajectory.

Language Command: "{language_command}"

Please evaluate the Current Video against the Candidate Next Videos (Index 0 to {len(candidate_files)-1}).
"""
    contents = [prompt_instruction, "\n[Current Video]:", current_video_file]
    for i, cand_file in enumerate(candidate_files):
        contents.extend([f"\n[Candidate Next Video {i}]:", cand_file])

    print("Requesting Gemini for evaluation...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FrameEvaluation,
            temperature=0.2
        )
    )

    print("\nRaw JSON Response:", response.text)
    result_dict = json.loads(response.text)
    print(f"==> The best video index chosen by VLM is: {result_dict['best_index']}")

    print("Cleaning up uploaded files...")
    client.files.delete(name=current_video_file.name)
    for cand_file in candidate_files:
        client.files.delete(name=cand_file.name)
        
    return result_dict['best_index']


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
    
    tasks = [
        {"port": 8010, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_0.png", "save": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/output_0.mp4"},
        {"port": 8011, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_1.png", "save": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/output_1.mp4"},
        {"port": 8012, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_2.png", "save": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/output_2.mp4"},
        {"port": 8013, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_3.png", "save": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/output_3.mp4"},
        {"port": 8014, "img": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/test_plan/plan_4.png", "save": "/home/zhiqil/workspace/fxz/openpi/third_party/libero/output_4.mp4"},
    ]
    
    print("Initializing parallel Wan2.2 video generation...")
    
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
        
    print("All candidate videos generated successfully!")
    generated_candidate_paths = [task["save"] for task in tasks]
    
    CURRENT_VIDEO_PATH = "/home/zhiqil/workspace/fxz/openpi/third_party/libero/rollout_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_ep0_success.mp4" 
    LANGUAGE_COMMAND = "put both the alphabet soup and the tomato sauce in the basket"
    
    try:
        best_index = evaluate_videos_with_gemini(
            current_video_path=CURRENT_VIDEO_PATH,
            candidate_paths=generated_candidate_paths,
            language_command=LANGUAGE_COMMAND
        )
        print(f"\nWorkflow complete! Proceeding with Plan {best_index}.")
    except Exception as e:
        print(f"Gemini evaluation failed: {e}")