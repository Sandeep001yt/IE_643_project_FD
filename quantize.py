from optimum.quanto import freeze, qfloat8, quantize
from diffusers import StableDiffusion3Pipeline
import torch
from optimum.quanto import Calibration, quantization_map  
from safetensors.torch import save_file
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import sys
import gc

pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
model = pipeline.transformer

quantize(model, weights=qint8, activations=qint8)

np.random.seed(0)
df = pd.read_csv("captions_unique.csv").sample(100)  
prompts = df['caption'].tolist()
calibration_samples = []
batch_size = 1

for i in tqdm(range(0, len(prompts), batch_size)):
    batch_prompts = prompts[i:i+batch_size]
    with torch.no_grad():
        for prompt in batch_prompts:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipeline.encode_prompt(
                    prompt=prompt,
                    prompt_2=None,
                    prompt_3=None,
                    device="cuda",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True
                    )        
            
            latents = torch.randn(
                1, 16, 64, 64,
                dtype=torch.float16,
                device="cuda"
            )
            
            timestep = torch.tensor(
                [random.randint(0, 1000)], 
                device="cuda"
            )
            
            calibration_samples.append({
                'hidden_states': latents,
                'encoder_hidden_states': prompt_embeds,
                'pooled_projections': pooled_prompt_embeds,
                'timestep': timestep
            })
            
            del prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds, latents, timestep
            torch.cuda.empty_cache()

print(f"Created {len(calibration_samples)} calibration samples")
print("Running calibration...")

with Calibration(momentum=0.9):
    for idx, sample in enumerate(tqdm(calibration_samples)):
        with torch.no_grad():
            _ = model(
                hidden_states=sample['hidden_states'],
                encoder_hidden_states=sample['encoder_hidden_states'],
                pooled_projections=sample['pooled_projections'],
                timestep=sample['timestep']
            )
        
        torch.cuda.empty_cache()

print("Freezing quantized model...")
freeze(model)

# Save the quantized model using quanto's save method
print("Saving quantized model...")
import os
save_dir = "sd3_transformer_quantized"
os.makedirs(save_dir, exist_ok=True)

# Save using torch.save with the quantization info
torch.save({
    'state_dict': model.state_dict(),
    'quantization_map': quantization_map(model)
}, os.path.join(save_dir, 'quantized_model.pt'))

print(f"Done! Quantized model saved to '{save_dir}/quantized_model.pt'")

# Optional: Test the quantized model
print("\nTesting quantized model...")
pipeline.transformer = model
test_prompt = "A cat wearing a wizard hat."
from time import time

with torch.no_grad():
    t = time()
    image = pipeline(
        prompt=test_prompt,
        num_inference_steps=20,
        guidance_scale=7.0,
        height=512,
        width=512
    ).images[0]
    t = time() - t
    
print(f"Inference time for quantized model: {t:.2f} seconds")
image.save("test_quantized_output.png")
print("Test image saved to 'test_quantized_output.png'")




