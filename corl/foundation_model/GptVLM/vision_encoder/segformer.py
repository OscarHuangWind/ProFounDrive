#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:38:24 2024

@author: oscar
"""
import cv2
import torch
import numpy as np

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

class SegEncoder():
    def __init__(self, device):
        super(SegEncoder, self).__init__()
    
        # segformer parameter
        self.device = device

        self.cityscapes_colormap = np.array([
            [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
            [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
            [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
            [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32], [  0,  0,  0],
            [128, 64, 64], [244, 35, 35], [ 70, 70, 70], [102,102,102], [190,153, 30],
            [153,153,  0], [250,170,153], [220,220,170], [107,142,135], [152,251,152],
            [ 70,130,100], [220, 20, 20], [255,  0,255], [  0,  0,  0]
        ], dtype=np.uint8)
    
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").cuda()
        self.feature_extractor.do_random_crop = False
        self.model.to(self.device)
        self.model.eval()
            
    def create_segmentation_overlay(self, segmentation_labels, colormap):
        height, width = segmentation_labels.shape
        segmentation_overlay = np.zeros((height, width, 3), dtype=np.uint8)
        for label in range(colormap.shape[0]):
            segmentation_overlay[segmentation_labels == label] = colormap[label]
    
        return segmentation_overlay
    
    def forward(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits # shape (batch_size, num_labels, height/4, width/4)
        
        segmentation_map = torch.argmax(logits.squeeze(), dim=0).detach().cpu().numpy()
        segmentation_overlay = self.create_segmentation_overlay(segmentation_map, self.cityscapes_colormap)
        segmentation_image = cv2.resize(segmentation_overlay, tuple((224, 224)))
        
        torch.cuda.empty_cache()

        return segmentation_image