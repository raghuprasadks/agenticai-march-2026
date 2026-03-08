# pip install diffusers transformers torch
from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu")  # Use "cuda" if you have a GPU

prompt = "A futuristic cityscape at sunset"

# Generate image
image = pipe(prompt).images[0]

# Save image
image.save("generated_image.png")
print("✅ Image generated and saved as generated_image.png")