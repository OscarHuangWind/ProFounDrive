#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:38:24 2024

@author: oscar
"""
import cv2
import torch
import numpy as np

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class SegEncoder():
    def __init__(self, device):
        super(SegEncoder, self).__init__()

        self.mapillary_vistas_colormap = np.asarray([
            [165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153], [180, 165, 180],
            [102, 102, 156], [102, 102, 156], [128, 64, 255], [140, 140, 200], [170, 170, 170],
            [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128], [110, 110, 110],
            [244, 35, 232], [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
            [255, 0, 0], [255, 0, 0], [255, 0, 0], [200, 128, 128], [255, 255, 255],
            [64, 170, 64], [128, 64, 64], [70, 130, 180], [255, 255, 255], [152, 251, 152],
            [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30], [0, 0, 0],
            [220, 220, 220], [170, 170, 170], [222, 40, 40], [100, 170, 30], [40, 40, 40],
            [33, 33, 33], [170, 170, 170], [0, 0, 142], [170, 170, 170], [210, 170, 100],
            [153, 153, 153], [128, 128, 128], [0, 0, 142], [250, 170, 30], [192, 192, 192],
            [220, 220, 0], [180, 165, 180], [119, 11, 32], [0, 0, 142], [0, 60, 100],
            [0, 0, 142], [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64], [0, 0, 110],
            [0, 0, 70], [0, 0, 192], [32, 32, 32], [0, 0, 0], [0, 0, 0]])

        self.feature_extractor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic").cuda()
        self.feature_extractor.do_random_crop = False
        self.model.eval()
            
    def create_segmentation_overlay(self, segmentation_labels, colormap):
        height, width = segmentation_labels.shape
        segmentation_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        for label in range(colormap.shape[0]):
            segmentation_overlay[segmentation_labels == label] = colormap[label]
    
        return segmentation_overlay
    
    def forward(self, image):
        torch.cuda.empty_cache()

        inputs = self.feature_extractor(images=image, return_tensors="pt") #.to(self.device)
        for key, data in inputs.items():
            inputs[key] = data.to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        segmentation_map = self.feature_extractor.post_process_semantic_segmentation(outputs)[0].data.cpu().numpy()
        segmentation_overlay = self.create_segmentation_overlay(segmentation_map, self.mapillary_vistas_colormap)
        segmentation_image = cv2.resize(segmentation_overlay, tuple((224, 224)))
        torch.cuda.empty_cache()

        return segmentation_image