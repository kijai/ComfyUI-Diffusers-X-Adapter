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

from .utils.single_file_utils import (create_scheduler_from_ldm, create_text_encoders_and_tokenizers_from_ldm, convert_ldm_vae_checkpoint, 
                                      convert_ldm_unet_checkpoint, create_text_encoder_from_ldm_clip_checkpoint, create_vae_diffusers_config, 
                                      create_diffusers_controlnet_model_from_ldm, create_unet_diffusers_config)
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
                "controlnet_image" : ("IMAGE",),
                "sd_1_5_checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
                "lora_checkpoint": (folder_paths.get_filename_list("loras"), ),
                "use_lora": ("BOOLEAN", {"default": False}),
                "width_sd1_5": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height_sd1_5": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "prompt_sd1_5": ("STRING", {"multiline": True, "default": "positive prompt sd1_5",}),

                "sdxl_checkpoint": (folder_paths.get_filename_list("checkpoints"), ),
                "width_sdxl": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height_sdxl": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "prompt_sdxl": ("STRING", {"multiline": True, "default": "positive prompt sdxl",}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "negative",}),
                "controlnet_name": (folder_paths.get_filename_list("controlnet"), ), 
                "guess_mode": ("BOOLEAN", {"default": False}),
                "control_guidance_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_guidance_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 4096, "step": 1}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "controlnet_condition_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "adapter_condition_scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                "adapter_guidance_start": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_xformers": ("BOOLEAN", {"default": False}),
                },
                "optional": {
                "latent_source_image" : ("IMAGE",),
                },             
            }
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "load_checkpoint"

    CATEGORY = "Diffusers-X-Adapter"

    def load_checkpoint(self, controlnet_image, prompt_sdxl, prompt_sd1_5, negative_prompt, use_xformers, sd_1_5_checkpoint, lora_checkpoint, use_lora, sdxl_checkpoint, 
                        controlnet_name, seed, steps, cfg, width_sd1_5, height_sd1_5, width_sdxl, height_sdxl,
                        adapter_condition_scale, adapter_guidance_start, controlnet_condition_scale, guess_mode, control_guidance_start, control_guidance_end, latent_source_image=None):
        torch.manual_seed(seed)
        device = comfy.model_management.get_torch_device()    
        dtype = torch.float16 if comfy.model_management.should_use_fp16() and not comfy.model_management.is_device_mps(device) else torch.float32

        control_image = controlnet_image.permute(0, 3, 1, 2)
        if latent_source_image is not None:
            latent_source_image = latent_source_image.permute(0, 3, 1, 2)

        model_path_sd1_5 = folder_paths.get_full_path("checkpoints", sd_1_5_checkpoint)
        lora_path = folder_paths.get_full_path("loras", lora_checkpoint)
        model_path_sdxl = folder_paths.get_full_path("checkpoints", sdxl_checkpoint)
        controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
        original_config = OmegaConf.load(os.path.join(script_directory, f"configs/v1-inference.yaml"))
        sdxl_original_config = OmegaConf.load(os.path.join(script_directory, f"configs/sd_xl_base.yaml"))
        controlnet_original_config = OmegaConf.load(os.path.join(script_directory, f"configs/control_v11p_sd15.yaml"))

        if not use_lora:
            self.current_lora = None

        if not hasattr(self, 'unet_sd1_5') or self.current_1_5_checkpoint != sd_1_5_checkpoint or self.current_lora != lora_checkpoint:
            comfy.model_management.soft_empty_cache()
            print("Loading SD_1_5 checkpoint: ", sd_1_5_checkpoint)
            self.current_1_5_checkpoint = sd_1_5_checkpoint
            self.current_lora = lora_checkpoint
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


            # 5. lora
            if use_lora:
                print("Loading LoRA: ", lora_checkpoint)
                self.lora_checkpoint = lora_checkpoint
                if lora_path.endswith(".safetensors"):
                    state_dict_lora = {}
                    with safe_open(lora_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            state_dict_lora[key] = f.get_tensor(key)
                elif lora_path.endswith(".ckpt"):
                    state_dict_lora = torch.load(lora_path, map_location="cpu")
                    while "state_dict" in state_dict_lora:
                        state_dict_lora = state_dict_lora["state_dict"]

                LORA_PREFIX_UNET = 'lora_unet'
                LORA_PREFIX_TEXT_ENCODER = 'lora_te'
                alpha = 1

                visited = []

                # directly update weight in diffusers model
                for key in state_dict_lora:

                    # it is suggested to print out the key, it usually will be something like below
                    # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

                    # as we have set the alpha beforehand, so just skip
                    if '.alpha' in key or key in visited:
                        continue

                    if 'text' in key:
                        layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER + '_')[-1].split('_')
                        curr_layer = self.text_encoder_sd1_5
                    else:
                        layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET + '_')[-1].split('_')
                        curr_layer = self.unet_sd1_5

                    # find the target layer
                    temp_name = layer_infos.pop(0)
                    while len(layer_infos) > -1:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(temp_name) > 0:
                                temp_name += '_' + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)

                    # org_forward(x) + lora_up(lora_down(x)) * multiplier
                    pair_keys = []
                    if 'lora_down' in key:
                        pair_keys.append(key.replace('lora_down', 'lora_up'))
                        pair_keys.append(key)
                    else:
                        pair_keys.append(key)
                        pair_keys.append(key.replace('lora_up', 'lora_down'))

                    # update weight
                    if len(state_dict_lora[pair_keys[0]].shape) == 4:
                        weight_up = state_dict_lora[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                        weight_down = state_dict_lora[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                    else:
                        weight_up = state_dict_lora[pair_keys[0]].to(torch.float32)
                        weight_down = state_dict_lora[pair_keys[1]].to(torch.float32)
                        curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

                    # update visited list
                    for item in pair_keys:
                        visited.append(item)
            else:
                self.current_lora = None
            del state_dict_sd1_5, converted_unet, converted_vae

        # load controlnet
        if not hasattr(self, 'controlnet') or self.current_controlnet_checkpoint != controlnet_name:
            print("Loading controlnet: ", controlnet_name)
            self.current_controlnet_checkpoint = controlnet_name
            #self.controlnet = ControlNetModel.from_single_file(controlnet_path)
            if controlnet_path.endswith(".safetensors"):
                state_dict_controlnet = {}
                with safe_open(controlnet_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict_controlnet[key] = f.get_tensor(key)
            else:
                state_dict_controlnet = torch.load(controlnet_path, map_location="cpu")
                while "state_dict" in state_dict_controlnet:
                    state_dict_controlnet = state_dict_controlnet["state_dict"]
            self.controlnet = create_diffusers_controlnet_model_from_ldm("ControlNet", controlnet_original_config, state_dict_controlnet)['controlnet']
            self.controlnet.to(dtype)

            del state_dict_controlnet

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
            comfy.model_management.soft_empty_cache()
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

            del state_dict_sdxl, converted_unet, converted_sdxl_stuff, converted_vae

            #xformers
            if use_xformers:
                self.unet_sd1_5.enable_xformers_memory_efficient_attention()
                self.unet_sdxl.enable_xformers_memory_efficient_attention()
                self.controlnet.enable_xformers_memory_efficient_attention()

            self.pipeline = StableDiffusionXLAdapterControlnetPipeline(
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
            controlnet=self.controlnet)

            self.pipeline.enable_model_cpu_offload()

            self.pipeline.scheduler_sd1_5.config.timestep_spacing = "leading"
            self.pipeline.unet.to(device=device, dtype=dtype)

        #run inference
        gen = Generator(device)
        gen.manual_seed(seed)
    
        img = \
            self.pipeline(prompt=prompt_sdxl, negative_prompt=negative_prompt, prompt_sd1_5=prompt_sd1_5,
                    width=width_sdxl, height=height_sdxl, height_sd1_5=height_sd1_5, width_sd1_5=width_sd1_5,
                    image=control_image,
                    num_inference_steps=steps, guidance_scale=cfg,
                    num_images_per_prompt=1, generator=gen,
                    controlnet_conditioning_scale=controlnet_condition_scale,
                    adapter_condition_scale=adapter_condition_scale,
                    adapter_guidance_start=adapter_guidance_start, guess_mode=guess_mode, control_guidance_start=control_guidance_start, control_guidance_end=control_guidance_end, source_img=latent_source_image).images[0]
        
        image_tensor = (img - img.min()) / (img.max() - img.min())
        if image_tensor.dim() ==  3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.permute(0,  2,  3,  1)
 
       
        return (image_tensor,)
        
NODE_CLASS_MAPPINGS = {
    "Diffusers_X_Adapter": Diffusers_X_Adapter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Diffusers_X_Adapter": "Diffusers_X_Adapter",
}        