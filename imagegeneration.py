from huggingface_hub import login

login()
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

torch.cuda.empty_cache()

# Initialize the Stable Diffusion Pipeline with NSFW content detection disabled
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", safety=None)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipe = pipe.to('cuda') if device == 'cuda' else pipe
c = 0
# Define the prompt
for j in range(3):
  prompt = input('Enter the object to be generated : ')
  for i in range(1):
    # Generate the image
    try:
        image = pipe(prompt).images[0]
        c += 1
        image.save('1'+".png")
        torch.cuda.empty_cache()
    except:
        break
