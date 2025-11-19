#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 20:03:11 2025

@author: kerencohen2
"""

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
# from typing import Callable, List, Optional, Sequence, Union
# from .api_wrappers import COCO

@DATASETS.register_module()
class CocoDatasetWithROI(CocoDataset):

    METAINFO = {
    'classes': ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat', ),
    'palette': [
            (255, 105, 180),   # pink       
            (135, 206, 250),   # light blue 
            (0, 255, 0),       # green       
            (160, 32, 240),    # purple      
            (255, 255, 0),     # yellow      
            (255, 0, 0),       # red       
        ]
    }
        
    def parse_data_info(self, raw_data_info):
        data_info = super().parse_data_info(raw_data_info)

        # raw_img_info is the dictionary from the COCO JSON "images" section
        raw_img_info = raw_data_info['raw_img_info']

        # Add roi_bbox if present
        if 'roi_bbox' in raw_img_info:
            data_info['roi_bbox'] = raw_img_info['roi_bbox']

        return data_info
