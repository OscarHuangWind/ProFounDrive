#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:59:16 2024

@author: oscar
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

# from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
# from transformers import SamModel, SamProcessor
# from transformers import SegformerModel, SegformerConfig, 
from transformers import pipeline
from PIL import Image
import cv2
import requests

from segformer import SegEncoder

# def show_mask(mask, ax, random_color=False):
#     if random_color:
#         color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#     else:
#         color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])segmentation_overlay
#     h, w = mask.shape[-2:]
#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     ax.imshow(mask_image)

# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
# feature_extractor.do_random_crop = False
# # feature_extractor.crop_size = (896, 896)
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").cuda()
# model.eval()

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic").cuda()
# feature_extractor.do_random_crop = False
model.to('cuda')
model.eval()

# seg_encoder = SegEncoder(device="cuda")
# model_sam = SamModel.from_pretrained("facebook/sam-vit-large").cuda()
# sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")

cityscapes_colormap = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32], [  0,  0,  0],
    [128, 64, 64], [244, 35, 35], [ 70, 70, 70], [102,102,102], [190,153, 30],
    [153,153,  0], [250,170,153], [220,220,170], [107,142,135], [152,251,152],
    [ 70,130,100], [220, 20, 20], [255,  0,255], [  0,  0,  0]
], dtype=np.uint8)

def maybe_autocast(dtype=torch.float16):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    # enable_autocast = self.device != torch.device("cpu")
    enable_autocast = True

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()

def create_segmentation_overlay(segmentation_labels, colormap):
    height, width = segmentation_labels.shape
    segmentation_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    for label in range(colormap.shape[0]):
        segmentation_overlay[segmentation_labels == label] = colormap[label]

    return segmentation_overlay

for i in range(500, 501):

    # with maybe_autocast():
    j = np.random.randint(100, 1000)

    # route_dir = '/media/oscar/Samsung_T5/CarlaData/InterFuser/town03_train/weather-debug/data/routes_town03_short_w0_12_01_01_37_34/rgb_front/0' + str(j) + '.jpg'
    route_dir = '/media/spyder/94714162-4d32-4b72-b5d5-74dc17149d9c/autonomous_driving/argoverse2/train/08734a1b-0289-3aa3-a6ba-8c7121521e26/sensors/cameras/ring_front_center/315971465849927217.jpg'
    image_i = np.array(route_dir).astype(np.string_)
    image = cv2.imread(str(image_i, encoding='utf-8'), cv2.IMREAD_COLOR)
    # image = cv2.resize(image, tuple((640, 480)))

    inputs = processor(images=image, return_tensors="pt")

    for key, data in inputs.items():
        inputs[key] = data.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs) 

    # url = 'https://www.freeway.com/knowledge-center/wp-content/uploads/sites/2/2022/08/new-driver-tips-city-vs-highway-driving-2.jpg'
    # image = Image.open(requests.get(url, stream=True).raw)
    
    # inputs = feature_extractor(images=image, return_tensors="pt").to('cuda')
    # # import pdb; pdb.set_trace()

    # with torch.no_grad():
    #     outputs = model(**inputs)
    # logits = outputs.logits # shape (batch_size, num_labels, height/4, width/4)
    
    # segmentation_map = torch.argmax(logits.squeeze(), dim=0).detach().cpu().numpy()
    # segmentation_overlay = create_segmentation_overlay(segmentation_map, cityscapes_colormap)
    # segmentation_overlay = cv2.resize(segmentation_overlay, tuple((224, 224)))

    # segmentation_image = seg_encoder.forward(image)
    
    predicted_semantic_map = processor.post_process_semantic_segmentation(outputs)[0]
    alpha = 0.6
    # overlay_image = cv2.addWeighted(image, alpha, segmentation_overlay, 1 - alpha, 0)
    plt.figure()
    plt.imshow(image)
    plt.figure()
    plt.imshow(predicted_semantic_map.data.cpu())

        # plt.figure()
        # plt.imshow(image)
        
        # plt.figure()
        # plt.imshow(segmentation_map)
    
        # inputs = sam_processor(image, return_tensors="pt").to("cuda")
        # for k,v in inputs.items():
        #     print(k,v.shape)
                
        # with torch.no_grad():
        #     outputs = model_sam(**inputs, multimask_output=False)
        
        # print(outputs.pred_masks.shape)
        
        # # apply sigmoid
        # medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # # convert soft mask to hard mask
        # medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
        # medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
        
        # plt.figure()
        # plt.imshow(np.array(image))
        
        # color = np.array([30/255, 144/255, 255/255, 0.6])
        # h, w = medsam_seg.shape[-2:]
        # mask_imageforward = medsam_seg.reshape(h, w, 1) * color.reshape(1, 1, -1)
        # plt.imshow(mask_image)

plt.show()

print('done')