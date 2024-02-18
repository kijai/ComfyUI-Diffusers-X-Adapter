import torch
import os
from diffusers import DPMSolverMultistepScheduler
from torch import Generator
from torchvision import transforms

from transformers import CLIPTokenizer, PretrainedConfig

from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler, ControlNetModel 

from .xadapter.model.unet_adapter import UNet2DConditionModel as UNet2DConditionModel_v2
from .xadapter.model.adapter import Adapter_XL
from .pipeline.pipeline_sd_xl_adapter_controlnet_img2img import StableDiffusionXLAdapterControlnetI2IPipeline
from .pipeline.pipeline_sd_xl_adapter_controlnet import StableDiffusionXLAdapterControlnetPipeline
from omegaconf import OmegaConf

from .utils.single_file_utils import (create_scheduler_from_ldm, create_text_encoders_and_tokenizers_from_ldm, convert_ldm_vae_checkpoint, convert_ldm_unet_checkpoint, create_text_encoder_from_ldm_clip_checkpoint, create_vae_diffusers_config, create_unet_diffusers_config)
from safetensors import safe_open

import comfy.model_management
import comfy.utils
import folder_paths

  
script_directory = os.path.dirname(os.path.abspath(__file__))

class Diffusers_X_Adapter:
    @classmethod
    def IS_CHANGED(s):
        return ""
    @classmethod
    def INPUT_TYPES(cls):

        return {"required":
                {
                "image" : ("IMAGE",),
                "sd_1_5_checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
                "width_sd1_5": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height_sd1_5": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "prompt_sd1_5": ("STRING", {"multiline": True, "default": "positive prompt sd1_5",}),

                "sdxl_checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
                "width_sdxl": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height_sdxl": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "prompt_sdxl": ("STRING", {"multiline": True, "default": "positive prompt sdxl",}),

                "controlnet_name": (folder_paths.get_filename_list("controlnet"), ), 
                
                "negative_prompt": ("STRING", {"multiline": True, "default": "negative",}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "controlnet_condition_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "adapter_condition_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.1}),
                "adapter_guidance_start": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.1}),
                "use_xformers": ("BOOLEAN", {"default": False}),
                },
                "optional": {
                "source_image" : ("IMAGE",),
                },             
            }
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "load_checkpoint"

    CATEGORY = "AD_MotionDirector"

    def load_checkpoint(self, image, prompt_sdxl, prompt_sd1_5, negative_prompt, use_xformers, sd_1_5_checkpoint, sdxl_checkpoint, 
                        controlnet_name, seed, steps, cfg, width_sd1_5, height_sd1_5, width_sdxl, height_sdxl,
                        adapter_condition_scale, adapter_guidance_start, controlnet_condition_scale, source_image=None):
        torch.manual_seed(seed)
        device = comfy.model_management.get_torch_device()    
        dtype = torch.float16 if comfy.model_management.should_use_fp16() and not comfy.model_management.is_device_mps(device) else torch.float32

        control_image = image.permute(0, 3, 1, 2)

        model_path_sd1_5 = folder_paths.get_full_path("checkpoints", sd_1_5_checkpoint)
        model_path_sdxl = folder_paths.get_full_path("checkpoints", sdxl_checkpoint)
        controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
        original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
        sdxl_original_config = OmegaConf.load(os.path.join(script_directory, f"configs/sd_xl_base.yaml"))

        if not hasattr(self, 'unet_sd1_5') or self.current_1_5_checkpoint != sd_1_5_checkpoint:
            print("Loading SD_1_5 checkpoint: ", sd_1_5_checkpoint)
            self.current_1_5_checkpoint = sd_1_5_checkpoint
            if model_path_sd1_5.endswith(".safetensors"):
                state_dict_sd1_5 = {}
                with safe_open(model_path_sd1_5, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict_sd1_5[key] = f.get_tensor(key)
            elif model_path_sd1_5.endswith(".ckpt"):
                state_dict_sd1_5 = torch.load(model_path_sd1_5, map_location="cpu")
                while "state_dict" in state_dict_sd1_5:
                    state_dict_sd1_5 = state_dict_sd1_5["state_dict"]

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(original_config, image_size=512)
            converted_vae = convert_ldm_vae_checkpoint(state_dict_sd1_5, converted_vae_config)
            self.vae_sd1_5 = AutoencoderKL(**converted_vae_config)
            self.vae_sd1_5.load_state_dict(converted_vae, strict=False)
            self.vae_sd1_5.to(dtype)

            # 2. unet
            converted_unet_config = create_unet_diffusers_config(original_config, image_size=512)
            converted_unet = convert_ldm_unet_checkpoint(state_dict_sd1_5, converted_unet_config)
            self.unet_sd1_5 = UNet2DConditionModel_v2(**converted_unet_config)
            self.unet_sd1_5.load_state_dict(converted_unet, strict=False)
            self.unet_sd1_5.to(dtype)

            # 3. text encoder and tokenizer
            self.tokenizer_sd1_5 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.text_encoder_sd1_5 = create_text_encoder_from_ldm_clip_checkpoint("openai/clip-vit-large-patch14",state_dict_sd1_5)
            self.text_encoder_sd1_5.to(dtype)

            # 4. scheduler
            self.scheduler_sd1_5 = create_scheduler_from_ldm("DPMSolverMultistepScheduler", original_config, state_dict_sd1_5, scheduler_type="ddim")['scheduler']

            del state_dict_sd1_5, converted_unet

        # load controlnet
        if not hasattr(self, 'controlnet') or self.current_controlnet_checkpoint != controlnet_name:
            print("Loading controlnet: ", controlnet_name)
            self.current_controlnet_checkpoint = controlnet_name
            self.controlnet = ControlNetModel.from_single_file(controlnet_path)
            self.controlnet.to(dtype)

        # load Adapter_XL
        if not hasattr(self, 'adapter'):
            adapter_checkpoint_path = os.path.join(script_directory, "checkpoints","X-Adapter")
            if not os.path.exists(adapter_checkpoint_path):
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo_id="Lingmin-Ran/X-Adapter", local_dir=adapter_checkpoint_path, local_dir_use_symlinks=False)
                except:
                    raise FileNotFoundError(f"No checkpoint directory found at {adapter_checkpoint_path}")
            adapter_ckpt = torch.load(os.path.join(adapter_checkpoint_path, "X_Adapter_v1.bin"))
            adapter = Adapter_XL()
            adapter.load_state_dict(adapter_ckpt)
            adapter.to(dtype)
    
        # load SDXL
        if not hasattr(self, 'unet_sdxl') or self.current_sdxl_checkpoint != sdxl_checkpoint:
            print("Loading SDXL checkpoint: ", sdxl_checkpoint)
            self.current_sdxl_checkpoint = sdxl_checkpoint
            if model_path_sdxl.endswith(".safetensors"):
                state_dict_sdxl = {}
                with safe_open(model_path_sdxl, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict_sdxl[key] = f.get_tensor(key)
            elif model_path_sdxl.endswith(".ckpt"):
                state_dict_sdxl = torch.load(model_path_sdxl, map_location="cpu")
                while "state_dict" in state_dict_sdxl:
                    state_dict_sdxl = state_dict_sdxl["state_dict"]

            # 1. vae
            converted_vae_config = create_vae_diffusers_config(sdxl_original_config, image_size=1024)
            converted_vae = convert_ldm_vae_checkpoint(state_dict_sdxl, converted_vae_config)
            self.vae_sdxl = AutoencoderKL(**converted_vae_config)
            self.vae_sdxl.load_state_dict(converted_vae, strict=False)
            self.vae_sdxl.to(dtype)

            # 2. unet
            converted_unet_config = create_unet_diffusers_config(sdxl_original_config, image_size=1024)
            converted_unet = convert_ldm_unet_checkpoint(state_dict_sdxl, converted_unet_config)
            self.unet_sdxl = UNet2DConditionModel_v2(**converted_unet_config)
            self.unet_sdxl.load_state_dict(converted_unet, strict=False)
            self.unet_sdxl.to(dtype)

            # 3. text encoders and tokenizers
            converted_sdxl_stuff = create_text_encoders_and_tokenizers_from_ldm(sdxl_original_config, state_dict_sdxl)
            self.tokenizer_one = converted_sdxl_stuff['tokenizer'] 
            self.sdxl_text_encoder = converted_sdxl_stuff['text_encoder']
            self.tokenizer_two = converted_sdxl_stuff['tokenizer_2']
            self.sdxl_text_encoder2 = converted_sdxl_stuff['text_encoder_2']
            self.sdxl_text_encoder.to(dtype)
            self.sdxl_text_encoder2.to(dtype)

            # 4. scheduler
            self.scheduler_sdxl = create_scheduler_from_ldm("DPMSolverMultistepScheduler", sdxl_original_config, state_dict_sdxl, scheduler_type="ddim",)['scheduler']

            del state_dict_sdxl, converted_unet


        #xformers
        if use_xformers:
            self.unet_sd1_5.enable_xformers_memory_efficient_attention()
            self.unet_sdxl.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()

        #print(type(scheduler_sd1_5))
        #print(type(scheduler_sdxl))

        torch.cuda.empty_cache()

        #run inference
        gen = Generator(device)
        gen.manual_seed(seed)
        
        if source_image is not None:
            pipe = StableDiffusionXLAdapterControlnetI2IPipeline(
                    vae=self.vae_sdxl,
                    text_encoder=self.sdxl_text_encoder,
                    text_encoder_2=self.sdxl_text_encoder2,
                    tokenizer=self.tokenizer_one,
                    tokenizer_2=self.tokenizer_two,
                    unet=self.unet_sdxl,
                    scheduler=self.scheduler_sdxl,
                    vae_sd1_5=self.vae_sd1_5,
                    text_encoder_sd1_5=self.text_encoder_sd1_5,
                    tokenizer_sd1_5=self.tokenizer_sd1_5,
                    unet_sd1_5=self.unet_sd1_5,
                    scheduler_sd1_5=self.scheduler_sd1_5,
                    adapter=adapter,
                    controlnet=self.controlnet
                )
        else:
                pipe = StableDiffusionXLAdapterControlnetPipeline(
                vae=self.vae_sdxl,
                text_encoder=self.sdxl_text_encoder,
                text_encoder_2=self.sdxl_text_encoder2,
                tokenizer=self.tokenizer_one,
                tokenizer_2=self.tokenizer_two,
                unet=self.unet_sdxl,
                scheduler=self.scheduler_sdxl,
                vae_sd1_5=self.vae_sd1_5,
                text_encoder_sd1_5=self.text_encoder_sd1_5,
                tokenizer_sd1_5=self.tokenizer_sd1_5,
                unet_sd1_5=self.unet_sd1_5,
                scheduler_sd1_5=self.scheduler_sd1_5,
                adapter=adapter,
                controlnet=self.controlnet
        )
        pipe.enable_model_cpu_offload()

        #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        #pipe.scheduler_sd1_5 = DPMSolverMultistepScheduler.from_config(pipe.scheduler_sd1_5.config)

        pipe.scheduler_sd1_5.config.timestep_spacing = "leading"
        pipe.unet.to(device=device, dtype=dtype)
        #pipe.unet.to(device=device, dtype=dtype, memory_format=torch.channels_last)
        iter_num = 1

        if source_image is not None:     
            for i in range(iter_num):
                img = \
                    pipe(prompt=prompt_sdxl, negative_prompt=negative_prompt, prompt_sd1_5=prompt_sd1_5,
                            width=width_sdxl, height=height_sdxl, height_sd1_5=height_sd1_5, width_sd1_5=width_sd1_5,
                            source_img=source_image, image=control_image,
                            num_inference_steps=steps, guidance_scale=cfg,
                            num_images_per_prompt=1, generator=gen,
                            controlnet_conditioning_scale=controlnet_condition_scale,
                            adapter_condition_scale=adapter_condition_scale,
                            adapter_guidance_start=adapter_guidance_start).images[0]
        else:
            for i in range(iter_num):
                img = \
                    pipe(prompt=prompt_sdxl, negative_prompt=negative_prompt, prompt_sd1_5=prompt_sd1_5,
                            width=width_sdxl, height=height_sdxl, height_sd1_5=height_sd1_5, width_sd1_5=width_sd1_5,
                            image=control_image,
                            num_inference_steps=steps, guidance_scale=cfg,
                            num_images_per_prompt=1, generator=gen,
                            controlnet_conditioning_scale=controlnet_condition_scale,
                            adapter_condition_scale=adapter_condition_scale,
                            adapter_guidance_start=adapter_guidance_start).images[0]
                        
        image_tensor = transforms.ToTensor()(img).unsqueeze(0).permute(0, 2, 3, 1)
        return (image_tensor,)
        
NODE_CLASS_MAPPINGS = {
    "Diffusers_X_Adapter": Diffusers_X_Adapter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Diffusers_X_Adapter": "Diffusers_X_Adapter",
}        