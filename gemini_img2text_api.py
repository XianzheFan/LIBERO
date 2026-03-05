import json
import base64
from pydantic import BaseModel
from google import genai
from google.genai import types

class FrameEvaluation(BaseModel):
    best_index: int

client = genai.Client(http_options={'api_version': 'v1alpha'})

current_frame_data = base64.b64decode("...") 
next_frame_0_data = base64.b64decode("...")
next_frame_1_data = base64.b64decode("...")
next_frame_2_data = base64.b64decode("...")
next_frame_3_data = base64.b64decode("...")
next_frame_4_data = base64.b64decode("...")

language_command = "Move the robotic arm to grasp the red can" 

prompt_instruction = f"""
You are an evaluation model in a robotic control system. 
Based on the [Current Frame] and the [Language Command], your task is to select the most accurate frame from the provided [Candidate Next Frames] that best follows the intended trajectory.

Language Command: "{language_command}"

Please evaluate the Current Frame against the 5 Candidate Next Frames (Index 0, 1, 2, 3 and 4).
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Content(
            parts=[
                # Instruction
                types.Part(text=prompt_instruction),
                
                # Current Frame
                types.Part(text="[Current Frame]:"),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=current_frame_data)),
                
                # Candidate 0
                types.Part(text="[Candidate Next Frame 0]:"),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=next_frame_0_data)),
                
                # Candidate 1
                types.Part(text="[Candidate Next Frame 1]:"),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=next_frame_1_data)),
                
                # Candidate 2
                types.Part(text="[Candidate Next Frame 2]:"),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=next_frame_2_data)),
                
                # Candidate 3
                types.Part(text="[Candidate Next Frame 3]:"),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=next_frame_3_data)),
                
                # Candidate 4
                types.Part(text="[Candidate Next Frame 4]:"),
                types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=next_frame_4_data)),
            ]
        )
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FrameEvaluation,
        temperature=0.2
    )
)

print("Raw JSON Response:", response.text)

result_dict = json.loads(response.text)
print(f"The best frame index chosen by VLM is: {result_dict['best_index']}")