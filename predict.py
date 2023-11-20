# Ignore line too long
# flake8: noqa: E501

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
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor
from PIL import Image

from dataset_and_utils import TokenEmbeddingsHandler

from gfpgan import GFPGANer
from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
import cv2
from compel import Compel
from transformers import CLIPFeatureExtractor
import insightface
import onnxruntime
from insightface.app import FaceAnalysis
from image_processing import (
    face_mask_google_mediapipe,
    crop_faces_to_square,
    paste_inpaint_into_original_image,
    get_head_mask,
)

SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

EMBEDDINGS = [(x.split(".")[0], "/embeddings/" + x) for x in os.listdir("/embeddings/")]
EMBEDDING_TOKENS = [x[0] for x in EMBEDDINGS]
EMBEDDING_PATHS = [x[1] for x in EMBEDDINGS]


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(
            config,
            use_karras_sigmas=True,
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
    def upscale_image_pil(self, img: Image.Image) -> Image.Image:
        weight = 0.5
        try:
            # Convert PIL Image to numpy array if necessary
            img = np.array(img)
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                # Convert grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            h, w = img.shape[0:2]
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            # Enhance the image using GFPGAN
            _, _, output = self.face_enhancer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=weight,
            )

            # Convert numpy array back to PIL Image
            output = Image.fromarray(output)

            return output

        except Exception as error:
            print("An exception occurred:", error)
            raise

    def build_controlnet_pipeline(self, pipeline_class, controlnet):
        pipe = pipeline_class(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            controlnet=controlnet,
        )

        pipe.to("cuda")

        return pipe

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.url = None

        if not os.path.exists("gfpgan/weights/realesr-general-x4v3.pth"):
            os.system(
                "wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P ./gfpgan/weights"
            )
        if not os.path.exists("gfpgan/weights/GFPGANv1.4.pth"):
            os.system(
                "wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P ./gfpgan/weights"
            )

        # background enhancer with RealESRGAN
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        model_path = "gfpgan/weights/realesr-general-x4v3.pth"
        half = True if torch.cuda.is_available() else False
        self.upsampler = RealESRGANer(
            scale=2,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
        )

        # Use GFPGAN for face enhancement
        self.face_enhancer = GFPGANer(
            model_path="gfpgan/weights/GFPGANv1.4.pth",
            upscale=1,
        )
        self.current_version = "v1.4"

        self.face_swapper = insightface.model_zoo.get_model(
            "cache/inswapper_128.onnx", providers=onnxruntime.get_available_providers()
        )
        self.face_analyser = FaceAnalysis(name="buffalo_l")
        self.face_analyser.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

    def get_face(self, img_data, image_type="target"):
        try:
            analysed = self.face_analyser.get(img_data)
            print(f"face num: {len(analysed)}")
            if len(analysed) == 0 and image_type == "source":
                msg = "no face"
                print(msg)
                raise Exception(msg)
            largest = max(
                analysed,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            )
            return largest
        except Exception as e:
            print(str(e))
            raise Exception(str(e))

    # Target image is image to paste into
    # Source image is image to take face from
    def swap_face(self, target_image: Path, source_image: Path) -> Image.Image:
        try:
            frame = cv2.imread(str(target_image))
            target_face = self.get_face(frame)
            source_face = self.get_face(
                cv2.imread(str(source_image)), image_type="source"
            )
            result = self.face_swapper.get(
                frame, target_face, source_face, paste_back=True
            )
            _, _, result = self.face_enhancer.enhance(result, paste_back=True)

            # Convert from BGR to RGB
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(result_rgb)
            return pil_image
        except Exception as e:
            print("FACESWAP ERROR", str(e))

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def download_zip_weights_python(self, url):
        """Download the model weights from the given URL"""
        print("Downloading weights...")

        if os.path.exists("weights"):
            shutil.rmtree("weights")
        os.makedirs("weights")

        import zipfile
        from io import BytesIO
        import urllib.request

        url = urllib.request.urlopen(url)
        with zipfile.ZipFile(BytesIO(url.read())) as zf:
            zf.extractall("weights")

    def load_weights(self, url):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading Safety pipeline...")

        if url == self.url:
            return

        start_time = time.time()
        self.download_zip_weights_python(url)
        print("Downloaded weights in {:.2f} seconds".format(time.time() - start_time))

        start_time = time.time()
        print("Loading SD pipeline...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "weights",
            safety_checker=None,
            feature_extractor=self.feature_extractor,
            torch_dtype=torch.float16,
        )

        print(EMBEDDING_TOKENS, EMBEDDING_PATHS)

        self.txt2img_pipe.load_textual_inversion(
            EMBEDDING_PATHS, token=EMBEDDING_TOKENS, local_files_only=True
        )

        self.txt2img_pipe.to("cuda")

        print("Loading SD img2img pipeline...")
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")

        print("Loading SD inpaint pipeline...")
        self.inpaint_pipe = StableDiffusionInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to("cuda")

        print("Loading controlnet...")

        controlnetModel = "lllyasviel/control_v11p_sd15_openpose"

        self.controlnet = ControlNetModel.from_pretrained(
            controlnetModel,
            torch_dtype=torch.float16,
            cache_dir="diffusers-cache",
            local_files_only=False,
        )

        print("Loaded pipelines in {:.2f} seconds".format(time.time() - start_time))

        self.txt2img_pipe.set_progress_bar_config(disable=True)
        self.img2img_pipe.set_progress_bar_config(disable=True)
        self.url = url

    @torch.inference_mode()
    def predict(
        self,
        weights: str = Input(
            description="LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="An photo of cjw man",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output. Supported embeddings: "
            + ", ".join(EMBEDDING_TOKENS),
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
            default=512,
        ),
        height: int = Input(
            description="Height of output image",
            default=512,
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
        should_swap_face: bool = Input(
            description="Should swap face",
            default=False,
        ),
        source_image: Path = Input(
            description="Source image for face swap",
            default=None,
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
            default=512,
        ),
        upscale_face: bool = Input(
            description="Upscale the face using GFPGAN",
            default=False,
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
            default="A photo of cjw man",
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

        weights = weights.replace(
            "https://replicate.delivery/pbxt/",
            "https://storage.googleapis.com/replicate-files/",
        )

        if weights is None:
            raise ValueError("No weights provided")
        self.load_weights(weights)

        # OOMs can leave vae in bad state
        if self.txt2img_pipe.vae.dtype == torch.float32:
            self.txt2img_pipe.vae.to(dtype=torch.float16)

        kwargs = {}
        if image and mask:
            print("inpainting mode")
            kwargs["image"] = self.load_image(image)
            kwargs["mask_image"] = self.load_image(mask)
            kwargs["strength"] = prompt_strength
            kwargs["width"] = width
            kwargs["height"] = height
        elif image:
            print("img2img mode")
            kwargs["image"] = self.load_image(image)
            kwargs["strength"] = prompt_strength
        else:
            print("txt2img mode")
            kwargs["width"] = width
            kwargs["height"] = height

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
                    StableDiffusionControlNetInpaintPipeline,
                    self.controlnet,
                )
            elif image:
                controlnet_args["control_image"] = pose_image
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionControlNetImg2ImgPipeline,
                    self.controlnet,
                )
            else:
                controlnet_args["image"] = pose_image
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionControlNetPipeline,
                    self.controlnet,
                )
        else:
            if image:
                pipe = self.img2img_pipe
            else:
                pipe = self.txt2img_pipe

        print(f"Prompt: {prompt}")

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        compel_proc = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            truncate_long_prompts=False,
        )

        common_args = {
            "prompt_embeds": compel_proc(prompt),
            "negative_prompt_embeds": compel_proc(negative_prompt),
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        first_pass = pipe(**common_args, **kwargs, **controlnet_args)

        self.output_paths = []
        for i, image in enumerate(first_pass.images):
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            self.output_paths.append(Path(output_path))

        swapped_images = []

        first_pass_done_images = first_pass.images

        # face swap
        if should_swap_face:
            if source_image:
                # Swap all faces in first pass images
                for i, image in enumerate(first_pass.images):
                    output_path = f"/tmp/out-faceswap-{i}.png"
                    swapped_image = self.swap_face(self.output_paths[i], source_image)
                    swapped_image.save(output_path)
                    self.output_paths.append(Path(output_path))
                    swapped_images.append(swapped_image)

                first_pass_done_images = swapped_images
            else:
                print("No source image provided, skipping face swap")

        face_masks = face_mask_google_mediapipe(
            first_pass_done_images, mask_blur_amount, 0
        )

        # Based on face detection, crop base image, mask image and pose image (if available)
        # to the face and save them to self.output_paths
        (
            cropped_face,
            cropped_mask,
            cropped_control,
            left_top,
            orig_size,
        ) = crop_faces_to_square(
            first_pass_done_images[0],
            face_masks[0],
            pose_image,
            face_padding,
            face_resize_to,
        )

        head_mask, head_mask_no_blur = get_head_mask(cropped_face, mask_blur_amount)

        # Add all to self.output_paths
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
                self.output_paths.append(Path(output_path))

        # fix_face
        if fix_face:
            # Run inpainting pipeline
            inpaint_generator = torch.Generator("cuda").manual_seed(seed)
            common_args = {
                "prompt": [inpaint_prompt] * num_outputs,
                "negative_prompt": [inpaint_negative_prompt] * num_outputs,
                "guidance_scale": inpaint_guidance_scale,
                "generator": inpaint_generator,
                "num_inference_steps": inpaint_num_inference_steps,
            }

            inpaint_kwargs = {}

            if upscale_face:
                upscaled_face = self.upscale_image_pil(cropped_face)
                # Add to self.output_paths
                output_path = f"/tmp/out-upscale-face.png"
                upscaled_face.save(output_path)
                self.output_paths.append(Path(output_path))
            else:
                upscaled_face = cropped_face

            inpaint_kwargs["image"] = upscaled_face
            inpaint_kwargs["mask_image"] = head_mask
            inpaint_kwargs["strength"] = inpaint_strength
            inpaint_kwargs["width"] = cropped_face.width
            inpaint_kwargs["height"] = cropped_face.height

            # Run inpainting pipeline
            if cropped_control:
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionControlNetInpaintPipeline,
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
                first_pass_done_images[0],
                left_top,
                inpaint_pass.images[0],
                orig_size,
                head_mask_no_blur,
            )

            # Save both inpaint result and pasted image to self.output_paths
            images_to_add = [
                inpaint_pass.images[0],
                pasted_image,
            ]

            for i, image in enumerate(images_to_add):
                output_path = f"/tmp/out-final-{i}.png"
                image.save(output_path)
                self.output_paths.append(Path(output_path))

        return self.output_paths
