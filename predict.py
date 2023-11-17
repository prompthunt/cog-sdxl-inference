import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from weights import WeightsDownloadCache

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

from dataset_and_utils import TokenEmbeddingsHandler


SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(
            config,
            use_karras_sigmas=True,
            euler_at_final=True,
            algorithm_type="sde-dpmsolver++",
        )


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "DPM++SDEKarras": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def load_trained_weights(self, weights, pipe):
        from no_init import no_init_or_tensor

        # weights can be a URLPath, which behaves in unexpected ways
        weights = str(weights)

        self.tuned_weights = weights

        local_weights_cache = self.weights_cache.ensure(weights)

        # load UNET
        print("Loading fine-tuned model")
        self.is_lora = False

        maybe_unet_path = os.path.join(local_weights_cache, "unet.safetensors")
        if not os.path.exists(maybe_unet_path):
            print("Does not have Unet. assume we are using LoRA")
            self.is_lora = True

        if not self.is_lora:
            print("Loading Unet")

            new_unet_params = load_file(
                os.path.join(local_weights_cache, "unet.safetensors")
            )
            # this should return _IncompatibleKeys(missing_keys=[...], unexpected_keys=[])
            pipe.unet.load_state_dict(new_unet_params, strict=False)

        else:
            print("Loading Unet LoRA")

            unet = pipe.unet

            tensors = load_file(os.path.join(local_weights_cache, "lora.safetensors"))

            unet_lora_attn_procs = {}
            name_rank_map = {}
            for tk, tv in tensors.items():
                # up is N, d
                if tk.endswith("up.weight"):
                    proc_name = ".".join(tk.split(".")[:-3])
                    r = tv.shape[1]
                    name_rank_map[proc_name] = r

            for name, attn_processor in unet.attn_processors.items():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else unet.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[
                        block_id
                    ]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                with no_init_or_tensor():
                    module = LoRAAttnProcessor2_0(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        rank=name_rank_map[name],
                    )
                unet_lora_attn_procs[name] = module.to("cuda", non_blocking=True)

            unet.set_attn_processor(unet_lora_attn_procs)
            unet.load_state_dict(tensors, strict=False)

        # load text
        handler = TokenEmbeddingsHandler(
            [pipe.text_encoder, pipe.text_encoder_2], [pipe.tokenizer, pipe.tokenizer_2]
        )
        handler.load_embeddings(os.path.join(local_weights_cache, "embeddings.pti"))

        # load params
        with open(os.path.join(local_weights_cache, "special_params.json"), "r") as f:
            params = json.load(f)
        self.token_map = params

        self.tuned_model = True

    def build_controlnet_pipeline(self, pipeline_class, controlnet):
        pipe = pipeline_class.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            controlnet=controlnet,
        )

        pipe.to("cuda")

        return pipe

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.weights_cache = WeightsDownloadCache()

        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SDXL_MODEL_CACHE):
            download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        # print("Loading SDXL img2img pipeline...")
        # self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        #     vae=self.txt2img_pipe.vae,
        #     text_encoder=self.txt2img_pipe.text_encoder,
        #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
        #     tokenizer=self.txt2img_pipe.tokenizer,
        #     tokenizer_2=self.txt2img_pipe.tokenizer_2,
        #     unet=self.txt2img_pipe.unet,
        #     scheduler=self.txt2img_pipe.scheduler,
        # )
        # self.img2img_pipe.to("cuda")

        # print("Loading SDXL inpaint pipeline...")
        # self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
        #     vae=self.txt2img_pipe.vae,
        #     text_encoder=self.txt2img_pipe.text_encoder,
        #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
        #     tokenizer=self.txt2img_pipe.tokenizer,
        #     tokenizer_2=self.txt2img_pipe.tokenizer_2,
        #     unet=self.txt2img_pipe.unet,
        #     scheduler=self.txt2img_pipe.scheduler,
        # )
        # self.inpaint_pipe.to("cuda")

        # print("Loading SDXL refiner pipeline...")
        # # FIXME(ja): should the vae/text_encoder_2 be loaded from SDXL always?
        # #            - in the case of fine-tuned SDXL should we still?
        # # FIXME(ja): if the answer to above is use VAE/Text_Encoder_2 from fine-tune
        # #            what does this imply about lora + refiner? does the refiner need to know about

        # if not os.path.exists(REFINER_MODEL_CACHE):
        #     download_weights(REFINER_URL, REFINER_MODEL_CACHE)

        # print("Loading refiner pipeline...")
        # self.refiner = DiffusionPipeline.from_pretrained(
        #     REFINER_MODEL_CACHE,
        #     text_encoder_2=self.txt2img_pipe.text_encoder_2,
        #     vae=self.txt2img_pipe.vae,
        #     torch_dtype=torch.float16,
        #     use_safetensors=True,
        #     variant="fp16",
        # )
        # self.refiner.to("cuda")
        # print("setup took: ", time.time() - start)
        # self.txt2img_pipe.__class__.encode_prompt = new_encode_prompt

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        lora_weights: str = Input(
            description="LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=1024,
        ),
        height: int = Input(
            description="Height of output image",
            default=1024,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "expert_ensemble_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        high_noise_frac: float = Input(
            description="For expert_ensemble_refiner, the fraction of noise to use",
            default=0.8,
            le=1.0,
            ge=0.0,
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=None,
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=True,
        ),
        pose_image: Path = Input(
            description="Pose image for controlnet",
            default=None,
        ),
        controlnet_conditioning_scale: float = Input(
            description="How strong the controlnet conditioning is",
            ge=0.0,
            le=4.0,
            default=0.75,
        ),
        controlnet_start: float = Input(
            description="When controlnet conditioning starts",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        controlnet_end: float = Input(
            description="When controlnet conditioning ends",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        fix_face: bool = Input(
            description="Fix the face in the image",
            default=False,
        ),
        mask_blur_amount: float = Input(
            description="Amount of blur to apply to the mask.", default=8.0
        ),
        face_padding: float = Input(
            description="Amount of padding (as percentage) to add to the face bounding box.",
            default=2,
        ),
        face_resize_to: int = Input(
            description="Resize the face bounding box to this size (in pixels).",
            default=1024,
        ),
        # ADD
        # inpaint_num_inference_steps
        # inpaint_guidance_scale
        # inpaint_strength
        # inpaint_lora_scale
        # inpaint_controlnet_conditioning_scale
        # inpaint_controlnet_start
        # inpaint_controlnet_end
        inpaint_prompt: str = Input(
            description="Input prompt",
            default="A photo of TOK",
        ),
        inpaint_negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="",
        ),
        inpaint_num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
        inpaint_guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=3
        ),
        inpaint_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.35,
        ),
        inpaint_lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        inpaint_controlnet_conditioning_scale: float = Input(
            description="How strong the controlnet conditioning is",
            ge=0.0,
            le=4.0,
            default=0.75,
        ),
        inpaint_controlnet_start: float = Input(
            description="When controlnet conditioning starts",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        inpaint_controlnet_end: float = Input(
            description="When controlnet conditioning ends",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.txt2img_pipe.to("cuda")

        self.txt2img_pipe.unload_lora_weights()

        if lora_weights:
            self.load_trained_weights(lora_weights, self.txt2img_pipe)

        print("Loading SDXL img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")

        print("Loading SDXL inpaint pipeline...")
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.inpaint_pipe.to("cuda")

        print("Loading controlnet model")
        self.controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16,
            cache_dir="/src/controlnet-cache",
        )
        self.controlnet.to("cuda")

        # OOMs can leave vae in bad state
        if self.txt2img_pipe.vae.dtype == torch.float32:
            self.txt2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        if image and mask:
            print("inpainting mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["mask_image"] = self.load_image(mask)
            sdxl_kwargs["strength"] = prompt_strength
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height
        elif image:
            print("img2img mode")
            sdxl_kwargs["image"] = self.load_image(image)
            sdxl_kwargs["strength"] = prompt_strength
        else:
            print("txt2img mode")
            sdxl_kwargs["width"] = width
            sdxl_kwargs["height"] = height

        controlnet_args = {}

        if pose_image:
            controlnet_args = {
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "control_guidance_start": controlnet_start,
                "control_guidance_end": controlnet_end,
            }
            pose_image = self.load_image(pose_image)
            if image and mask:
                controlnet_args["control_image"] = pose_image
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionXLControlNetInpaintPipeline,
                    self.controlnet,
                )
            elif image:
                controlnet_args["control_image"] = pose_image
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionXLControlNetImg2ImgPipeline,
                    self.controlnet,
                )
            else:
                controlnet_args["image"] = pose_image
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionXLControlNetPipeline,
                    self.controlnet,
                )

        else:
            if image and mask:
                pipe = self.inpaint_pipe
            elif image:
                pipe = self.img2img_pipe
            else:
                pipe = self.txt2img_pipe

        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        print(f"Prompt: {prompt}")

        if refine == "expert_ensemble_refiner":
            sdxl_kwargs["output_type"] = "latent"
            sdxl_kwargs["denoising_end"] = high_noise_frac
        elif refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None
            self.refiner.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        first_pass = pipe(**common_args, **sdxl_kwargs, **controlnet_args)

        if not apply_watermark:
            pipe.watermark = watermark_cache
            self.refiner.watermark = watermark_cache

        output_paths = []
        for i, image in enumerate(first_pass.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        # fix_face
        if fix_face:
            from image_processing import (
                face_mask_google_mediapipe,
                crop_faces_to_square,
                paste_inpaint_into_original_image,
                get_head_mask,
            )

            face_masks = face_mask_google_mediapipe(
                first_pass.images, mask_blur_amount, 0
            )

            # Based on face detection, crop base image, mask image and pose image (if available)
            # to the face and save them to output_paths
            (
                cropped_face,
                cropped_mask,
                cropped_control,
                left_top,
                orig_size,
            ) = crop_faces_to_square(
                first_pass.images[0],
                face_masks[0],
                pose_image,
                face_padding,
                face_resize_to,
            )

            head_mask = get_head_mask(cropped_face, mask_blur_amount)

            # Add all to output_paths
            images_to_add = [
                cropped_face,
                cropped_mask,
                cropped_control,
                head_mask,
            ]
            for i, image in enumerate(images_to_add):
                # If image is image and exists
                if image and image.size:
                    output_path = f"/tmp/out-processing-{i}.png"
                    image.save(output_path)
                    output_paths.append(Path(output_path))

            inpaint_generator = torch.Generator("cuda").manual_seed(seed)
            common_args = {
                "prompt": [inpaint_prompt] * num_outputs,
                "negative_prompt": [inpaint_negative_prompt] * num_outputs,
                "guidance_scale": inpaint_guidance_scale,
                "generator": inpaint_generator,
                "num_inference_steps": inpaint_num_inference_steps,
            }

            inpaint_kwargs = {}

            inpaint_kwargs["image"] = cropped_face
            inpaint_kwargs["mask_image"] = head_mask
            inpaint_kwargs["strength"] = inpaint_strength
            inpaint_kwargs["width"] = cropped_face.width
            inpaint_kwargs["height"] = cropped_face.height

            if self.is_lora:
                inpaint_kwargs["cross_attention_kwargs"] = {"scale": inpaint_lora_scale}

            # Run inpainting pipeline
            if cropped_control:
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionXLControlNetInpaintPipeline,
                    self.controlnet,
                )
                controlnet_args = {
                    "controlnet_conditioning_scale": inpaint_controlnet_conditioning_scale,
                    "control_guidance_start": inpaint_controlnet_start,
                    "control_guidance_end": inpaint_controlnet_end,
                    "control_image": cropped_control,
                }
            else:
                pipe = self.inpaint_pipe

            inpaint_pass = pipe(**common_args, **inpaint_kwargs, **controlnet_args)

            # Paste inpainted face back into original image
            pasted_image = paste_inpaint_into_original_image(
                first_pass.images[0],
                left_top,
                inpaint_pass.images[0],
                orig_size,
                head_mask,
            )

            # Save both inpaint result and pasted image to output_paths
            images_to_add = [
                inpaint_pass.images[0],
                pasted_image,
            ]

            for i, image in enumerate(images_to_add):
                output_path = f"/tmp/out-final-{i}.png"
                image.save(output_path)
                output_paths.append(Path(output_path))

        return output_paths
