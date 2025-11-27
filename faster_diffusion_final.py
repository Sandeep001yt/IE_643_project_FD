import torch
from diffusers import StableDiffusion3Pipeline
from typing import Set, Optional, List, Dict, Union
import numpy as np
import pandas as pd
# import streamlit as st
import time
import urllib.parse

class SplitMMDiTSD3:
    def __init__(
            self,
            pipe: StableDiffusion3Pipeline,
            encoder_blocks: int = 12,
            key_timesteps = [20, 18, 15, 10, 5],
            num_inference_steps: int = 20,
            guidance_scale: float = 7.5,
            noise_factor: float = 0
    ):
        self.pipe = pipe
        self.encoder_blocks = encoder_blocks
        self.key_timesteps = key_timesteps
        self.guidance_scale = guidance_scale
        self.transformer = pipe.transformer
        self.scheduler = pipe.scheduler
        self.vae = pipe.vae
        self.num_inference_steps = num_inference_steps
        self.noise_factor = noise_factor
        self.total_blocks = len(self.pipe.transformer.transformer_blocks)
        self.decoder_blocks = self.total_blocks - self.encoder_blocks
    
    def run_encoder(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            temb: torch.Tensor,
    ):
        encoder_output = hidden_states
        for i in range(self.encoder_blocks):
            encoder_hidden_states,encoder_output = self.transformer.transformer_blocks[i](
                hidden_states=encoder_output,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,)
            
        return encoder_hidden_states,encoder_output

    def run_decoder_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ): 
        decoder_output = hidden_states
        dec_hidden_states=encoder_hidden_states
        
        for i in range(self.encoder_blocks, self.total_blocks):
            dec_hidden_states,decoder_output = self.transformer.transformer_blocks[i](
                hidden_states=decoder_output,
                encoder_hidden_states=dec_hidden_states,
                temb=temb,
            )
        return dec_hidden_states,decoder_output

    def generate_with_key_timesteps(
            self,
            hidden_states: torch.Tensor,
            t: torch.Tensor,
            encoder_hidden_states: torch.Tensor, # Encoder outputs from text encoder
            temb: torch.Tensor,
            current_latents: torch.Tensor,
            height: Optional[int] = 512,
            width: Optional[int] = 512
    ):  
        # Run encoder blocks
        encoder_hidden_states,encoder_output = self.run_encoder(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
        )

        # Run decoder blocks
        dec_hidden_states=encoder_hidden_states
        dec_hidden_states,decoder_output = self.run_decoder_blocks(
            hidden_states=encoder_output,
            encoder_hidden_states=dec_hidden_states,
            temb=temb,
        )
        hidden_states = self.transformer.norm_out(decoder_output, temb)
        hidden_states = self.transformer.proj_out(hidden_states)

        # Unpatchify
        patch_size = self.transformer.config.patch_size
        height = height // patch_size
        width = width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.transformer.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        noise_pred = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.transformer.out_channels, height * patch_size, width * patch_size)
        )

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)           

        # Compute the previous noisy sample x_t -> x_t-1
        prev_latents = self.scheduler.step(noise_pred, t, current_latents)[0]
        return prev_latents, encoder_output,encoder_hidden_states
        

    def run_faster_diffusion(
            self,
            prompt: str,
            negative_prompt: Optional[str] = None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            guidance_scale: float = 7.5,
            generator: Optional[torch.Generator] = None,
            **kwargs
    ):
        device = self.pipe._execution_device
        dtype = self.transformer.dtype

        # Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=guidance_scale > 1.0,
            device=device,
        )

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        # Setup scheduler
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self.key_timesteps_new = [timesteps[self.num_inference_steps - i] for i in self.key_timesteps]
        current_latents = latents
        if guidance_scale > 1.0:
            encoder_hidden_states_cfg = torch.cat([negative_prompt_embeds, prompt_embeds],dim=0)
            pooled_projections_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds],dim=0)
        else:
            encoder_hidden_states_cfg = prompt_embeds
            pooled_projections_cfg = pooled_prompt_embeds

        encoder_hidden_states_processed = self.transformer.context_embedder(encoder_hidden_states_cfg)
        encoder_output = None
        final_encoder_hidden_state = None
        latent_model_input = None
        height_latent, width_latent = current_latents.shape[-2:]
        non_keystep_list=[]
        i=0
        while i < len(timesteps):
            t= timesteps[i]
            if t in self.key_timesteps_new:
                latent_model_input = torch.cat([current_latents] * 2) if guidance_scale > 1.0 else current_latents
                timestep_tensor = None
                timestep_tensor = t.expand(latent_model_input.shape[0])

                # Positional embedding
                hidden_states = self.transformer.pos_embed(latent_model_input) 

                # Get time text embedding
                temb = self.transformer.time_text_embed(
                            timestep=timestep_tensor,
                            pooled_projection=pooled_projections_cfg
                        )
                
                current_latents, encoder_output,final_encoder_hidden_state = self.generate_with_key_timesteps(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states_processed,
                    temb=temb,
                    current_latents=current_latents,
                    t = t,
                    height=height_latent,
                    width=width_latent
                )
                i+=1
            else:
                parallel_data = []
                batch_encoder_output = []
                batch_tembs = []
                batch_encoder_hidden_states = []
                while  i < len(timesteps) and timesteps[i] not in self.key_timesteps_new:
                    t= timesteps[i]
                    timestep_tensor = t.expand(latent_model_input.shape[0])
                    temb = self.transformer.time_text_embed(
                                timestep=timestep_tensor,
                                pooled_projection=pooled_projections_cfg
                            )
                    parallel_data.append({
                        'idx': i,
                        't': t,
                        'temb': temb
                    })
                    i+=1
                for data in parallel_data:
                    batch_encoder_output.append(encoder_output + torch.randn_like(encoder_output) * self.noise_factor)
                    batch_tembs.append(data['temb'])
                    batch_encoder_hidden_states.append(final_encoder_hidden_state)
                num_parallel = len(parallel_data)
                # Stack for batch processing
                batched_encoder_outputs = torch.cat(batch_encoder_output, dim=0)
                batched_tembs = torch.cat(batch_tembs, dim=0)
                batched_encoder_hidden_states = torch.cat(batch_encoder_hidden_states, dim=0)
                batched_decoder_output = self.run_decoder_blocks(
                    hidden_states=batched_encoder_outputs,
                    encoder_hidden_states=batched_encoder_hidden_states,
                    temb=batched_tembs,
                )[1]

                batched_hidden_states = self.transformer.norm_out(batched_decoder_output, batched_tembs)
                batched_hidden_states = self.transformer.proj_out(batched_hidden_states)

                # Unpatchify
                patch_size = self.transformer.config.patch_size
                h = height_latent // patch_size
                w = width_latent // patch_size
                batch_size_per_step = 2 if guidance_scale > 1.0 else 1
                batched_decoder_outputs = batched_hidden_states.reshape(
                    batch_size_per_step * num_parallel, h, w, patch_size, patch_size, self.transformer.out_channels
                )
                batched_decoder_outputs = batched_decoder_outputs.permute(0, 5, 1, 3, 2, 4)
                batched_decoder_outputs = batched_decoder_outputs.reshape(
                    batch_size_per_step * num_parallel, self.transformer.out_channels, height_latent, width_latent
                )
                
                # Split outputs for each timestep
                parallel_outputs = batched_decoder_outputs.chunk(num_parallel, dim=0)

                for idx in range(len(parallel_data)):
                    t_step = timesteps[idx]
                    noise_pred = parallel_outputs[idx]
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    current_latents = self.scheduler.step(noise_pred, t_step, current_latents, return_dict=False)[0]

        # Decode latents to image
        current_latent = (current_latents/ self.vae.config.scaling_factor) + self.vae.config.shift_factor
        images = self.vae.decode(current_latent, return_dict=False)[0]
        images = self.pipe.image_processor.postprocess(images, output_type="pil")
        return type('Result', (), {'images': images})()
        




def main():
    from time import time
    import os
    # pipe = StableDiffusion3Pipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-3-medium-diffusers",
    #     torch_dtype=torch.float16
    # )
    # pipe.to("cuda")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        text_encoder_3=None,
        tokenizer_3=None,      # also set tokenizer_3 to None
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # with torch.no_grad():
    #     prompt = "a photo of a cat holding a sign that says hello world"
    #     fd= SplitMMDiTSD3(
    #         pipe,
    #         key_timesteps= [20,19,18,15,10,8,5,4],
    #         encoder_blocks=10, 
    #         num_inference_steps=20, 
    #         guidance_scale=7.5
    #         )
    #     t = time()
    #     image = fd.run_faster_diffusion(
    #         prompt= prompt
    #     ).images[0]
    #     t = time()-t
    #     image.save(f'demo_image.png')
    #     print(f'Time Taken: {t}')
    print("DONE")
    df = pd.read_csv("dataset/captions_unique.csv").iloc[:5000]
    time_taken = []
    # key_timesteps_list = [[20,17,10,4,3], [20,18,16,8,5], [20,18,15,10,5]]
    # encoder_blocks_list = [6, 10]
    # key_timesteps_list = [[20,17,10,4,3], [20,18,16,8,5], [20,18,15,10,5]]
    # encoder_blocks_list = [12, 18]
    key_timesteps_list= [[20, 19, 18 ,15, 10, 8, 5, 4]]
    encoder_blocks_list = [10]

    for encoder_blocks in encoder_blocks_list:
        for key_timesteps in key_timesteps_list:
            fd= SplitMMDiTSD3(
                pipe,
                key_timesteps= key_timesteps,
                encoder_blocks=encoder_blocks, 
                num_inference_steps=20, 
                guidance_scale=7.5, 
                noise_factor=0
                )
            file = f"dataset/fd_key_{key_timesteps}_enc_{encoder_blocks}_new"
            # os.makedirs(file, exist_ok= False)
            df['time'] = 0
            with torch.no_grad():
                for i in df.iterrows():
                    t = time()
                    image = fd.run_faster_diffusion(
                        prompt= i[1]['caption']
                    ).images[0]
                    t = time()-t
                    image.save(f"{file}/{i[1]['image']}")
                    print(f'Time Taken: {t}')
                    df.loc[i[0], 'time'] = t
                    df.to_csv(f"{file}.csv")

# if __name__ == "__main__":
#     main()

