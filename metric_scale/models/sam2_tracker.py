#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def show_mask(mask, image, random_color=False, borders=True, color=None):
    if color is not None:
        pass
    elif random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    # Remove singleton dimensions
    mask = mask.squeeze()
    
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    
    # Create mask image with color
    mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
    mask_colored[mask == 1] = (color[:3] * 255).astype(np.uint8)
    
    # Apply mask to image
    image = cv2.addWeighted(image, 1.0, mask_colored, 0.6, 0)
    
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=2)
    
    return image

def show_box(box, image, color=(0, 255, 0), label=None):
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[2]), int(box[3])
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tx = x0
        ty = max(th + baseline + 2, y0 - 4)
        cv2.rectangle(
            image,
            (tx, ty - th - baseline),
            (tx + tw, ty + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(image, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image

def show_masks(image, masks, scores, box_coords=None, labels=None, borders=True, save_path=None):
    if not save_path:
        return
    # Convert PIL image to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        # Assuming image is already a numpy array in RGB format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Predefined colors for different masks
    colors = [
        np.array([30/255, 144/255, 255/255, 0.6]),    # Blue
        np.array([255/255, 99/255, 71/255, 0.6]),      # Tomato
        np.array([144/255, 238/255, 144/255, 0.6]),    # LightGreen
        np.array([255/255, 215/255, 0/255, 0.6]),      # Gold
        np.array([138/255, 43/255, 226/255, 0.6]),     # BlueViolet
        np.array([255/255, 165/255, 0/255, 0.6]),      # Orange
        np.array([0/255, 255/255, 255/255, 0.6]),      # Cyan
        np.array([255/255, 192/255, 203/255, 0.6]),    # Pink
        np.array([32/255, 178/255, 170/255, 0.6]),     # LightSeaGreen
        np.array([218/255, 112/255, 214/255, 0.6])     # Orchid
    ]
    
    # Check masks shape
    for i, (mask, score, box) in enumerate(zip(masks, scores, box_coords)):
        # Get color for this mask
        color = colors[i % len(colors)]
        # Apply mask with specific color
        image = show_mask(mask, image, borders=borders, color=color)
        score_val = score[0] if isinstance(score, (list, np.ndarray)) else score
        label_prefix = labels[i] if labels is not None and i < len(labels) else None
        label = f"{label_prefix} {score_val:.2f}" if label_prefix else f"{score_val:.2f}"
        image = show_box(box, image, color=(color[:3] * 255).astype(np.uint8).tolist(), label=label)
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        cv2.putText(image, f"{score_val:.2f}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
        
class SAM2Tracker:
    """Simplified SAM2 Tracker for basic segmentation"""

    def __init__(self, model_cfg, sam2_checkpoint, device=None):
        """Initialize SAM2 Video Predictor"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

        # Use bfloat16 for better performance if supported
        if self.device_type == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            device_index = torch.device(self.device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            if torch.cuda.get_device_properties(device_index).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def segment(self, image_list, image_boxes_list, output_dir=None, labels_batch=None):
        """
        Segment objects in a frame using box prompts
        
        Args:
            image_list: List of images to segment
            image_boxes_list: List of box prompts for each image, each containing box coordinates (x1, y1, x2, y2)
            output_dir: Directory to save segmented images
        
        Returns:
            masks: Segmentation masks for the objects
        """
        img_batch = image_list
        boxes_batch = image_boxes_list
        self.predictor.set_image_batch(img_batch)
        masks_batch, scores_batch, _ = self.predictor.predict_batch(
            None,
            None, 
            box_batch=boxes_batch, 
            multimask_output=False
        )
        if output_dir:
            for i, (image, boxes, masks) in enumerate(zip(img_batch, boxes_batch, masks_batch)):
                labels = None if labels_batch is None or i >= len(labels_batch) else labels_batch[i]
                show_masks(
                    image,
                    masks,
                    scores_batch[i],
                    box_coords=boxes,
                    labels=labels,
                    save_path=output_dir,
                )
            
        return masks_batch
