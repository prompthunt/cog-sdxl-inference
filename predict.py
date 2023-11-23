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
    UniPCMultistepScheduler,
    LMSDiscreteScheduler,
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
from compel import Compel, DiffusersTextualInversionManager
from controlnet_aux import OpenposeDetector
from transformers import CLIPFeatureExtractor
import insightface
import onnxruntime
from insightface.app import FaceAnalysis


SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

EMBEDDINGS = [
    (x.split(".")[0], "./embeddings/" + x) for x in os.listdir("./embeddings/")
]
EMBEDDING_TOKENS = [x[0] for x in EMBEDDINGS]
EMBEDDING_PATHS = [x[1] for x in EMBEDDINGS]


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(
            config,
            use_karras_sigmas=True,
        )


class KarrasDPMPP:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(
            config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
        )


def make_scheduler(name, config):
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "HeunDiscrete": HeunDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        "UniPCMultistep": UniPCMultistepScheduler.from_config(config),
        "KarrasDPM": KarrasDPM.from_config(config),
        "DPM++SDEKarras": KarrasDPMPP.from_config(config),
    }[name]


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

    def process_control(self, control_image):
        if control_image is None:
            return None

        return self.openpose(control_image)

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        print("Loading pose...")
        self.openpose = OpenposeDetector.from_pretrained(
            "lllyasviel/ControlNet", cache_dir="diffusers-cache"
        )

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
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=self.upsampler,
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
        ).to("cuda")

        self.txt2img_pipe.load_textual_inversion(
            EMBEDDING_PATHS, token=EMBEDDING_TOKENS, local_files_only=True
        )

        # https://huggingface.co/docs/diffusers/using-diffusers/weighted_prompts#textual-inversion
        # https://pypi.org/project/compel/ check out Textual Inversion support

        textual_inversion_manager = DiffusersTextualInversionManager(self.txt2img_pipe)

        self.compel_proc = Compel(
            tokenizer=self.txt2img_pipe.tokenizer,
            text_encoder=self.txt2img_pipe.text_encoder,
            textual_inversion_manager=textual_inversion_manager,
            truncate_long_prompts=False,
        )

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
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            torch_dtype=torch.float16,
            cache_dir="diffusers-cache",
        )

        print("Loading controlnet txt2img...")
        self.cnet_txt2img_pipe = StableDiffusionControlNetPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            controlnet=controlnet,
        ).to("cuda")

        print("Loading controlnet img2img...")
        self.cnet_img2img_pipe = StableDiffusionControlNetImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            controlnet=controlnet,
        )

        print("Loaded pipelines in {:.2f} seconds".format(time.time() - start_time))

        # self.txt2img_pipe.set_progress_bar_config(disable=True)
        # self.img2img_pipe.set_progress_bar_config(disable=True)
        # self.inpaint_pipe.set_progress_bar_config(disable=True)
        self.url = url

    @torch.inference_mode()
    def predict(
        self,
        weights: str = Input(
            description="Weights url",
            default=None,
        ),
        control_image: Path = Input(
            description="Optional Image to use for guidance based on posenet",
            default=None,
        ),
        pose_image: Path = Input(
            description="Direct Pose image to use for guidance based on posenet, if available, ignores control_image",
            default=None,
        ),
        image: Path = Input(
            description="Optional Image to use for img2img guidance",
            default=None,
        ),
        mask: Path = Input(
            description="Optional Mask to use for legacy inpainting",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="photo of cjw person",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output. Supported embeddings: "
            + ", ".join(EMBEDDING_TOKENS),
            default="",
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
            le=40,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "DPMSolverMultistep",
                "HeunDiscrete",
                "K_EULER_ANCESTRAL",
                "K_EULER",
                "KLMS",
                "PNDM",
                "UniPCMultistep",
                "KarrasDPM",
                "DPM++SDEKarras",
            ],
            description="Choose a scheduler.",
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
        show_debug_images: bool = Input(
            description="Show debug images",
            default=False,
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

        if image:
            image = self.load_image(image)
        if pose_image:
            control_image = self.load_image(pose_image)
        elif control_image:
            control_image = self.load_image(control_image)
            control_image = self.process_control(control_image)
        if mask:
            mask = self.load_image(mask)

        kwargs = {}
        if control_image and mask:
            raise ValueError("Cannot use controlnet and inpainting at the same time")
        elif control_image and image:
            print("Using ControlNet img2img")
            pipe = self.cnet_img2img_pipe
            extra_kwargs = {
                "controlnet_conditioning_image": control_image,
                "image": image,
                "strength": prompt_strength,
            }
        elif control_image:
            print("Using ControlNet txt2img")
            pipe = self.cnet_txt2img_pipe
            extra_kwargs = {
                "image": control_image,
                "width": width,
                "height": height,
            }
        elif image and mask:
            print("Using inpaint pipeline")
            pipe = self.inpainting_pipe
            # FIXME(ja): prompt/negative_prompt are sent to the inpainting pipeline
            # because it doesn't support prompt_embeds/negative_prompt_embeds
            extra_kwargs = {
                "image": image,
                "mask_image": mask,
                "strength": prompt_strength,
            }
        elif image:
            print("Using img2img pipeline")
            pipe = self.img2img_pipe
            extra_kwargs = {
                "image": image,
                "strength": prompt_strength,
            }
        else:
            print("Using txt2img pipeline")
            pipe = self.txt2img_pipe
            extra_kwargs = {
                "width": width,
                "height": height,
            }

        print(f"Prompt: {prompt}")
        print(f"Negative Prompt: {negative_prompt}")

        if prompt:
            conditioning = self.compel_proc.build_conditioning_tensor(prompt)
            if not negative_prompt:
                negative_prompt = ""  # it's necessary to create an empty prompt - it can also be very long, if you want
            negative_conditioning = self.compel_proc.build_conditioning_tensor(
                negative_prompt
            )
            [
                prompt_embeds,
                negative_prompt_embeds,
            ] = self.compel_proc.pad_conditioning_tensors_to_same_length(
                [conditioning, negative_conditioning]
            )

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(this_seed)
            output = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )

            output_path = f"/tmp/seed-{this_seed}.png"
            output.images[0].save(output_path)
            path_to_output = Path(output_path)
            # If show_debug_images or is no face swap
            if show_debug_images or not should_swap_face:
                yield path_to_output

            if should_swap_face:
                if source_image:
                    # Swap all faces in first pass images
                    output_path = f"/tmp/seed-swapped-{this_seed}.png"
                    swapped_image = self.swap_face(path_to_output, source_image)
                    swapped_image.save(output_path)
                    path_to_output = Path(output_path)
                    yield path_to_output
                else:
                    print("No source image provided, skipping face swap")
