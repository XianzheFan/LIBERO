from google import genai
from PIL import Image

client = genai.Client()

prompt = ("""Look at the robotic arm workspace in the input image. The drawn trajectory represents a candidate future motion: the green dot is the start, and the red dot is the end destination. Please imagine and edit the image to show the state immediately after this motion completes.
Follow these conditional physics rules:
1. Move the Robotic Arm: The center of the end-effector (gripper) must be perfectly aligned with the position of the red dot in the final image.
2. Object Interaction Logic: Observe the relationship between the green dot (start position) and any objects in the scene.
- IF the green dot is currently positioned near an object in a way that implies a grasp, contact, or if the gripper is already holding it: Move that specific object along the trajectory so that it is still being held or manipulated by the gripper at the red dot.
- IF the green dot and the predicted path are in free space, not touching any objects: Execute the movement as pure free-space motion. No objects should be moved.
- Do not change the open/closed state of the gripper unless it is necessary for physical plausibility at the destination (e.g., placing).
3. Environment Stability: Keep all other background elements, lighting, camera angle, and non-interacted objects exactly identical to the original scene.""")

image = Image.open("/path/to/image.png")

response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[prompt, image],
)

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = part.as_image()
        image.save("generated_image.png")