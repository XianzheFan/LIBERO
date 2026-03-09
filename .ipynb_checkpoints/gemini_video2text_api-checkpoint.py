import json
import time
from pydantic import BaseModel
from google import genai
from google.genai import types

class FrameEvaluation(BaseModel):
    best_index: int

client = genai.Client(http_options={'api_version': 'v1alpha'})

print("Uploading videos to Google GenAI...")
current_video_file = client.files.upload(file="path/to/current_video.mp4")
next_video_0_file = client.files.upload(file="path/to/next_video_0.mp4")
next_video_1_file = client.files.upload(file="path/to/next_video_1.mp4")
next_video_2_file = client.files.upload(file="path/to/next_video_2.mp4")
next_video_3_file = client.files.upload(file="path/to/next_video_3.mp4")
next_video_4_file = client.files.upload(file="path/to/next_video_4.mp4")

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

wait_for_files_active(current_video_file, next_video_0_file, next_video_1_file) 

language_command = "Move the robotic arm to grasp the red can" 

prompt_instruction = f"""
You are an evaluation model in a robotic control system. 
Based on the [Current Video] and the [Language Command], your task is to select the most accurate video from the provided [Candidate Next Videos] that best follows the intended trajectory.

Language Command: "{language_command}"

Please evaluate the Current Video against the Candidate Next Videos (Index 0, 1, 2, 3 and 4).
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        prompt_instruction,
        
        "\n[Current Video]:",
        current_video_file,
        
        "\n[Candidate Next Video 0]:",
        next_video_0_file,
        
        "\n[Candidate Next Video 1]:",
        next_video_1_file,
        
        "\n[Candidate Next Video 2]:",
        next_video_2_file,
        
        "\n[Candidate Next Video 3]:",
        next_video_3_file,
        
        "\n[Candidate Next Video 4]:",
        next_video_4_file,
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FrameEvaluation,
        temperature=0.2
    )
)

print("Raw JSON Response:", response.text)

result_dict = json.loads(response.text)
print(f"The best video index chosen by VLM is: {result_dict['best_index']}")

client.files.delete(name=current_video_file.name)
client.files.delete(name=next_video_0_file.name)
client.files.delete(name=next_video_1_file.name)
client.files.delete(name=next_video_2_file.name)
client.files.delete(name=next_video_3_file.name)
client.files.delete(name=next_video_4_file.name)
