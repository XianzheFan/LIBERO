import json
import time
from pydantic import BaseModel
from google import genai
from google.genai import types
from typing import List

class FrameEvaluation(BaseModel):
    reasoning: str
    score: float
    status: str

client = genai.Client(http_options={'api_version': 'v1alpha'})

print("Uploading videos to Google GenAI...")
current_video_file = client.files.upload(file="data/libero/libero_10_videos/rollout_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_ep0_success/complete_video.mp4")

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

wait_for_files_active(current_video_file) 

language_command = "" 

prompt_instruction = f"""You are a top-tier robot action evaluation expert responsible for constructing a Dense Value Function for an RL model. Based on the provided video sequence (including the past 17s of history), please evaluate the robot's state **over the most recent 5s** and provide a **Value Score** between **0.00** and **1.00**.
Rigorous Scoring Scale:
- 0.00 - 0.20 (Disengaged/Failure State): The robot is not in contact with the target object, is moving in the wrong direction, or has just committed a serious destructive error (e.g., knocking something over, dropping an item).
- 0.20 - 0.40 (Approach State): The robot's end-effector is moving correctly toward the target object and preparing for contact, but stable interaction has not yet occurred.
- 0.40 - 0.60 (Initial Interaction State): Successful contact or grasping of the target object has been achieved, but the core task logic has not yet begun (e.g., has not yet started moving or placing the object).
- 0.60 - 0.80 (Critical Execution State): The core task is being executed smoothly and is only one step away from the final goal state.
- 0.80 - 1.00 (Completion State): The task has been successfully accomplished.
Please output strictly in **JSON array format** (without any additional explanatory text). Include reasoning (justification for the score based on the scale), score (final score, rounded to two decimal places) and status. **Example Format:** [{{"reasoning": "The end-effector is approaching the target but has not yet made contact.", "score": 0.35, "status": "Approach State"}}]
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        prompt_instruction,
        
        "\n[Current Video]:",
        current_video_file,
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[FrameEvaluation],
        temperature=0.0
    )
)

print("Raw JSON Response:", response.text)
result_dict = json.loads(response.text)
client.files.delete(name=current_video_file.name)
# Raw JSON Response: [{"reasoning": "The robot has successfully placed the black object into the organizer and released it, completing the intended task.", "score": 1.0, "status": "Completion State"}]
