from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import mmdet

from mmdet.apis import DetInferencer 

import cv2
import matplotlib.pyplot as plt
import random
from pycocotools.coco import COCO
import os
import numpy as np
import time 
from configs.models_registry import MODEL_ZOO
from configs.config import MODEL_KEY
from utils.roi_cropper import ROICropper

def get_model_config(model_key: str):
    if model_key not in MODEL_ZOO:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODEL_ZOO.keys())}")
    return MODEL_ZOO[model_key]["model_name"], MODEL_ZOO[model_key]["checkpoint"]


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = f'{mmdet.__version__}+{get_git_hash()[:7]}'
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')

    data_dir = 'data/Dataset'
    ann_path = os.path.join(data_dir, 'annotations')
    train_ann_path = os.path.join(ann_path, 'annotations_train.json')
    test_path = os.path.join(data_dir, 'test2017')
    train_path = os.path.join(data_dir, 'train2017')
    val_path = os.path.join(data_dir, 'val2017')
    
    # Initialize COCO
    coco = COCO(train_ann_path)
    
    # Get Crop car roi pretrained model
    model_name, checkpoint = get_model_config(MODEL_KEY)

    print(f'Using model {MODEL_KEY}')
    # model_name = 'yolox_s_8x8_300e_coco'#'rtmdet_tiny_8xb32-300e_coco'
    # checkpoint = 'checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'#'checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    
    # --- Initialize and run inference ---
    inferencer = DetInferencer(model_name, checkpoint)
    class_names = inferencer.model.dataset_meta['classes']

    cropper = ROICropper(class_names)

    k = 10
    
    for _ in range(k):
        # Pick random image
        img_id = random.choice(coco.getImgIds())
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(train_path, img_info['file_name'])
        output_path = os.path.join('output', f'id_{str(img_id)}')
        
        os.makedirs(output_path, exist_ok=True)

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_manipulated = image.copy()
        
        # Load annotations for this image
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        
        # Overlay segmentations and bounding boxes
        for ann in anns:
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw segmentation mask (if exists)
            if 'segmentation' in ann and ann['segmentation']:
                mask = coco.annToMask(ann)
                image_manipulated[mask == 1] = image_manipulated[mask == 1] * 0.5 + np.array(color) * 0.5  # blend
            
            # Draw bounding box
            x, y, w, h = ann['bbox']
            cv2.rectangle(image_manipulated, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            
            # Label with category name
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            cv2.putText(image_manipulated, cat_name, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"{img_info['file_name']}  |  ID: {img_id}")
        plt.axis('off')
        plt.savefig(os.path.join(output_path, 'image.jpg'))
    
        plt.show() 
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image_manipulated)
        plt.title(f"{img_info['file_name']}  |  ID: {img_id}")
        plt.axis('off')
        plt.savefig(os.path.join(output_path, 'image_manipulated.jpg'))
    
        plt.show()      
            
    
        result = inferencer(image, out_dir=output_path)
        
        result_roi = result['visualization'][0]
        pred = result['predictions'][0]
    
        bboxes = np.array(pred['bboxes'])
        scores = np.array(pred['scores'])
        labels = np.array(pred['labels'])
    
        plt.figure(figsize=(8, 8))
        plt.imshow(result_roi)
        plt.title(f"result_roi {img_info['file_name']}  |  ID: {img_id}")
        plt.axis('off')
        plt.savefig(os.path.join(output_path, 'result_roi.jpg'))
    
        plt.show()   
        
        # Prepare cropper
        img_h, img_w = image.shape[:2]

        roi, cls, class_name, score = cropper.extract_roi(bboxes, scores, labels, img_w, img_h)

    
    stats = cropper.get_statistics()
    print(stats)

    print('done')
    