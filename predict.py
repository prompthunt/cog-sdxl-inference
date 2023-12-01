# Ignore line too long
# flake8: noqa: E501

import uuid
import hashlib
import os
import shutil
import subprocess
import time
from typing import List

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
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
    AutoencoderKL,
    # StableDiffusionControlNetInpaintPipeline,
)
from cloudflare import upload_to_cloudflare, get_watermarked_image


from diffusers.utils import load_image
from PIL import Image


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
from codeformer.app import inference_app


def resize_for_condition_image(input_image: Image, k: float):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


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

    def has_faces(self, img_data: Path):
        """
        Checks if the given image data contains any faces.
        Returns True if faces are found, otherwise False.
        """
        frame = cv2.imread(str(img_data))
        try:
            analysed = self.face_analyser.get(frame)
            return len(analysed) > 0
        except Exception as e:
            print(str(e))
            return False

    def filter_images_with_faces(self, source_images: List[Path]):
        """
        Filters the given array of source images.
        Returns a new array containing only those images with faces.
        """
        images_with_faces = []
        for img in source_images:
            if self.has_faces(img):
                images_with_faces.append(img)

        return images_with_faces

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
        print("Downloading weights...")

        # Generate a hash value from the URL
        url_hash = hashlib.md5(url.encode()).hexdigest()
        hash_folder = os.path.join("weights", url_hash)

        # Check if the folder already exists and has contents
        if os.path.exists(hash_folder) and os.listdir(hash_folder):
            print("Weights already downloaded.")
            return hash_folder
        else:
            # Remove the folder if it exists and is empty, then recreate it
            if os.path.exists(hash_folder):
                shutil.rmtree(hash_folder)
            os.makedirs(hash_folder)

            import zipfile
            from io import BytesIO
            import urllib.request

            # Download and extract the weights
            url_response = urllib.request.urlopen(url)
            with zipfile.ZipFile(BytesIO(url_response.read())) as zf:
                zf.extractall(hash_folder)
            print("Weights downloaded and saved in:", hash_folder)

        return hash_folder

    def load_weights(self, url, use_new_vae=False):
        """Load the model into memory to make running multiple predictions efficient"""

        # Release resources held by existing pipeline objects
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.inpaint_pipe = None
        self.cnet_tile_pipe = None
        self.cnet_txt2img_pipe = None
        self.cnet_img2img_pipe = None
        torch.cuda.empty_cache()  # Clear GPU cache if using CUDA

        start_time = time.time()
        weights_folder = self.download_zip_weights_python(
            url
        )  # This now returns the folder path
        print("Downloaded weights in {:.2f} seconds".format(time.time() - start_time))

        start_time = time.time()
        print("Loading SD pipeline...")

        if use_new_vae:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

            self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                weights_folder,
                safety_checker=None,
                feature_extractor=self.feature_extractor,
                vae=vae,
                torch_dtype=torch.float16,
            ).to("cuda")
        else:
            self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                weights_folder,
                safety_checker=None,
                feature_extractor=self.feature_extractor,
                torch_dtype=torch.float16,
            ).to("cuda")

        # Embedding path is current path / embeddings
        EMBEDDING_PATHS = [
            os.path.join(os.getcwd(), "embeddings", x)
            for x in os.listdir(os.path.join(os.getcwd(), "embeddings"))
        ]

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

        print("Loading pose controlnet...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose",
            torch_dtype=torch.float16,
        )
        # print("Loading tile controlnet...")
        # controlnet_tile = ControlNetModel.from_pretrained(
        #     "lllyasviel/control_v11f1e_sd15_tile",
        #     torch_dtype=torch.float16,
        # )

        # print("Loading tile pipeline...")
        # self.cnet_tile_pipe = StableDiffusionControlNetImg2ImgPipeline(
        #     vae=self.txt2img_pipe.vae,
        #     text_encoder=self.txt2img_pipe.text_encoder,
        #     tokenizer=self.txt2img_pipe.tokenizer,
        #     unet=self.txt2img_pipe.unet,
        #     scheduler=self.txt2img_pipe.scheduler,
        #     safety_checker=self.txt2img_pipe.safety_checker,
        #     feature_extractor=self.txt2img_pipe.feature_extractor,
        #     controlnet=controlnet_tile,
        # ).to("cuda")

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
        prompt_2: str = Input(
            description="Input prompt",
            default=None,
        ),
        prompt_3: str = Input(
            description="Input prompt",
            default=None,
        ),
        prompt_4: str = Input(
            description="Input prompt",
            default=None,
        ),
        pose_image_2: Path = Input(
            description="Direct Pose image to use for guidance based on posenet, if available, ignores control_image",
            default=None,
        ),
        pose_image_3: Path = Input(
            description="Direct Pose image to use for guidance based on posenet, if available, ignores control_image",
            default=None,
        ),
        pose_image_4: Path = Input(
            description="Direct Pose image to use for guidance based on posenet, if available, ignores control_image",
            default=None,
        ),
        source_image_2: Path = Input(
            description="Source image for face swap",
            default=None,
        ),
        source_image_3: Path = Input(
            description="Source image for face swap",
            default=None,
        ),
        source_image_4: Path = Input(
            description="Source image for face swap",
            default=None,
        ),
        upscale_final_image: bool = Input(
            description="Upscale final image",
            default=True,
        ),
        upscale_final_size: int = Input(
            description="Upscale final size multiplier",
            default=4,
        ),
        upscale_fidelity: float = Input(
            description="Upscale codeformer fidelity",
            default=0.7,
        ),
        upscale_background_enhance: bool = Input(
            description="Upscale background enhance",
            default=True,
        ),
        upscale_face_upsample: bool = Input(
            description="Upscale face upsample",
            default=True,
        ),
        use_new_vae: bool = Input(
            description="Use new vae",
            default=False,
        ),
        second_pass_strength: float = Input(
            description="Second pass strength",
            default=0.8,
        ),
        second_pass_guidance_scale: float = Input(
            description="Second pass guidance scale",
            default=3,
        ),
        second_pass_steps: int = Input(
            description="Second pass steps",
            default=50,
        ),
        cf_acc_id: str = Input(
            description="Cloudflare account ID",
            default=None,
        ),
        cf_api_key: str = Input(
            description="Cloudflare API key",
            default=None,
        ),
    ) -> List[str]:
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
        self.load_weights(weights, use_new_vae=use_new_vae)

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

        if pose_image_2:
            control_image_2 = self.load_image(pose_image_2)
        else:
            control_image_2 = None
        if pose_image_3:
            control_image_3 = self.load_image(pose_image_3)
        else:
            control_image_3 = None
        if pose_image_4:
            control_image_4 = self.load_image(pose_image_4)
        else:
            control_image_4 = None

        kwargs = {}
        if control_image and mask:
            raise ValueError("Cannot use controlnet and inpainting at the same time")
        elif control_image and image:
            print("Using ControlNet img2img")
            pipe = self.cnet_img2img_pipe
            extra_kwargs = {
                "control_image": control_image,
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

        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        prompts = [prompt, prompt_2, prompt_3, prompt_4]
        # Remove non existent prompts
        prompts = [x for x in prompts if x]

        control_images = [
            control_image,
            control_image_2,
            control_image_3,
            control_image_4,
        ]
        # Remove non existent control images
        control_images = [x for x in control_images if x]

        source_images = [
            source_image,
            source_image_2,
            source_image_3,
            source_image_4,
        ]
        # Remove non existent source images
        source_images = [x for x in source_images if x]

        initial_output_images = []

        for idx in range(num_outputs):
            this_seed = seed + idx
            generator = torch.Generator("cuda").manual_seed(this_seed)
            print(f"Prompt: {prompt}")
            print(f"Negative Prompt: {negative_prompt}")

            # Pick a prompt round robin
            prompt = prompts[idx % len(prompts)]

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

            if control_image and image:
                control_image = control_images[idx % len(control_images)]
                extra_kwargs["control_image"] = control_image
            elif control_image:
                control_image = control_images[idx % len(control_images)]
                extra_kwargs["image"] = control_image

            output = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )

            initial_output_images.append(output.images[0])

        # Resize all initial images by 1.5
        resized_initial_output_images = []
        for idx, output in enumerate(initial_output_images):
            resized_image = resize_for_condition_image(output, 1.5)
            resized_initial_output_images.append(resized_image)

        # Resize condition images by 1.5
        resized_control_images = []
        for idx, control_image in enumerate(control_images):
            resized_image = resize_for_condition_image(control_image, 1.5)
            resized_control_images.append(resized_image)

        second_pass_images = []
        pipe = self.cnet_img2img_pipe
        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        # Run second passes
        for idx, resized_initital_image in enumerate(resized_initial_output_images):
            # Get new seed and generator
            this_seed = seed + idx + 1000
            generator = torch.Generator("cuda").manual_seed(this_seed)

            # Pick a prompt round robin
            prompt = prompts[idx % len(prompts)]

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

            control_image = control_images[idx % len(control_images)]

            output = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=second_pass_guidance_scale,
                generator=generator,
                num_inference_steps=second_pass_steps,
                control_image=control_image,
                image=resized_initital_image,
                strength=second_pass_strength,
            )

            second_pass_images.append(output.images[0])

        # Swap faces
        swapped_faces_images_paths = []
        for idx, second_pass_image in enumerate(second_pass_images):
            source_image_to_use = source_images[idx % len(source_images)]

            second_pass_image_path = f"/tmp/second-pass-{idx}.png"
            second_pass_image.save(second_pass_image_path)
            second_pass_image_path = Path(second_pass_image_path)

            output_path = f"/tmp/seed-swapped-{idx}.png"
            swapped_image = self.swap_face(second_pass_image_path, source_image_to_use)
            # Save swapped image and add path to swapped_faces_images
            swapped_image.save(output_path)
            swapped_image_path = Path(output_path)
            swapped_faces_images_paths.append(swapped_image_path)

            # If show_debug_images or no upscale
            # if show_debug_images or not upscale_final_image:
            #     yield path_to_output

            # output_path = f"/tmp/seed-{this_seed}.png"
            # output.images[0].save(output_path)
            # path_to_output = Path(output_path)
            # If show_debug_images or (no swap and no upscale)
            # if show_debug_images or (not should_swap_face and not upscale_final_image):
            #     yield path_to_output

            # if should_swap_face:
            #     source_image_to_use = source_images[idx % len(source_images)]
            #     if source_image:
            #         # Swap all faces in first pass images
            #         output_path = f"/tmp/seed-swapped-{this_seed}.png"
            #         swapped_image = self.swap_face(path_to_output, source_image_to_use)
            #         swapped_image.save(output_path)
            #         path_to_output = Path(output_path)
            #         # If show_debug_images or no upscale
            #         # if show_debug_images or not upscale_final_image:
            #         #     yield path_to_output
            #     else:
            #         print("No source image provided, skipping face swap")

            # Second pass

            # output_path = f"/tmp/seed-second-{second_seed}.png"
            # output.images[0].save(output_path)
            # path_to_output = Path(output_path)
            # If show_debug_images or (no swap and no upscale)
            # if show_debug_images or (not should_swap_face and not upscale_final_image):
            #     yield path_to_output

        # Upscale all swapped images
        for idx, swapped_faces_image_path in enumerate(swapped_faces_images_paths):
            upscaled_image_path = inference_app(
                image=swapped_faces_image_path,
                background_enhance=upscale_background_enhance,
                face_upsample=upscale_face_upsample,
                upscale=upscale_final_size,
                codeformer_fidelity=upscale_fidelity,
            )
            path_to_output = Path(upscaled_image_path)
            # yield path_to_output

            if cf_acc_id and cf_api_key:
                print("Uploading to Cloudflare...")
                try:
                    # uuid
                    # image to use is upscaled_image_path if exists, else output_path
                    image_to_use = (
                        upscaled_image_path if upscale_final_image else output_path
                    )
                    id = str(uuid.uuid4())
                    cf_url = upload_to_cloudflare(
                        id,
                        str(image_to_use),
                        cf_acc_id,
                        cf_api_key,
                    )
                    print("Uploaded to Cloudflare:", cf_url)
                    yield cf_url

                    # # Watermark the image
                    # watermarked_image_url = get_watermarked_image(
                    #     cf_url,
                    #     576,
                    #     cf_acc_id,
                    #     cf_api_key,
                    # )
                    # print("Watermarked image:", watermarked_image_url)

                    # Return the watermarked image
                    # yield watermarked_image_url
                except Exception as e:
                    print("Failed to upload to Cloudflare", str(e))
