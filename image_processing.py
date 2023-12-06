import numpy as np
import torch
from PIL import Image, ImageFilter
import head_segmentation.segmentation_pipeline as seg_pipeline
import cv2


def get_head_mask(
    head_image: Image.Image,
    blur_amount: float = 8.0,
) -> Image.Image:
    # Setting up the device for the segmentation pipeline
    device = torch.device("cuda")

    # Initialize the segmentation pipeline
    segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=device)

    # Convert the image to a numpy array
    image_np = np.array(head_image)

    # Predict the segmentation map
    segmentation_map = segmentation_pipeline.predict(image_np)

    # Create a mask where the head is white and the rest is black
    head_mask = np.where(segmentation_map == 1, 255, 0).astype(np.uint8)

    # Convert the mask to a 3-channel image (white head, black background)
    segmented_region = cv2.cvtColor(head_mask, cv2.COLOR_GRAY2RGB)

    # Convert the result to a PIL image
    pil_image = Image.fromarray(segmented_region)

    # Apply blur amount
    pil_image_blur = pil_image.filter(ImageFilter.GaussianBlur(blur_amount))
    return pil_image_blur
