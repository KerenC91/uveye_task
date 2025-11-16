from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import mmdet

from mmdet.apis import DetInferencer 
import sys
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
import torch 
import json


def get_model_config(model_key: str):
    if model_key not in MODEL_ZOO:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODEL_ZOO.keys())}")
    return MODEL_ZOO[model_key]["model_name"], MODEL_ZOO[model_key]["checkpoint"]


# random.seed(42)


if __name__ == '__main__':
    # Define paths
    data_dir = 'data/Dataset'
    if not os.path.exists(data_dir):
        print("Error: invalid paths.")
        sys.exit(1)  # stop the program
    
    ## config ##
    dataset = 'val' # train, test, val
    use_fixed_ann = True
    ############
    
    ann_dir_path = os.path.join(data_dir, 'annotations')
    data_dir_path = os.path.join(data_dir, f'{dataset}2017')
    
    if (not os.path.exists(ann_dir_path) or
        not os.path.exists(data_dir_path)):
        print("Error: invalid paths.")
        sys.exit(1)  # stop the program
    
    s_fixed_ann = "_fixed" if use_fixed_ann else ""
    ann_path = os.path.join(ann_dir_path, f'annotations_{dataset}{s_fixed_ann}.json')
    
    if not os.path.exists(ann_path):
        print("Error: invalid paths.")
        sys.exit(1)  # stop the program
        
    print(f'Dataset: {dataset}')        
    print(f'Input annotation file: {ann_path}')
    
    # Initialize COCO dataset
    coco = COCO(ann_path)
    
    # Load the full annotation JSON so we can modify and save it
    with open(ann_path, "r") as f:
        coco_json = json.load(f)
    
    # Build a mapping image_id --> image dict
    image_dict = {img["id"]: img for img in coco_json["images"]}


    # Get pretrained model for cropping car roi 
    model_name, checkpoint = get_model_config(MODEL_KEY)

    print(f'Using model {MODEL_KEY}')

    # Initialize and run inference
    inferencer = DetInferencer(model_name, checkpoint)
    class_names = inferencer.model.dataset_meta['classes']

    cropper = ROICropper(class_names)

    # Randomly choose data_size image IDs (no duplicates)
    data_size = 0
    img_ids = coco.getImgIds()
    chosen_ids = random.sample(img_ids, data_size)
    subset_ids = chosen_ids if data_size else img_ids

    print(f'Chosen ids: {"all" if data_size == 0 else subset_ids}')
    
    times = []
    updated_annotations = {}

    #define th
    th = 0.3
    
    for img_id in subset_ids:
        img_info = coco.loadImgs(img_id)[0]        
        img_path = os.path.join(data_dir_path, img_info['file_name'])
        output_path = os.path.join('output', f'{str(img_id)}')
        
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
        # plt.figure(figsize=(8, 8))
        # plt.imshow(image)
        # plt.title(f"{img_info['file_name']}  |  ID: {img_id}")
        # plt.axis('off')
        # plt.savefig(os.path.join(output_path, 'image.jpg'))
    
        # plt.show() 
        
        # plt.figure(figsize=(8, 8))
        # plt.imshow(image_manipulated)
        # plt.title(f"{img_info['file_name']}  |  ID: {img_id}")
        # plt.axis('off')
        # plt.savefig(os.path.join(output_path, 'image_manipulated.jpg'))
    
        # plt.show()      
        plt.imsave(os.path.join(output_path, 'image_manipulated.jpg'), image_manipulated)

        torch.cuda.synchronize()
        start_time = time.time() 
        
        result = inferencer(image, return_vis=True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        times.append(end_time - start_time)

        
        result_roi = result['visualization'][0]
        pred = result['predictions'][0]
    
        bboxes = np.array(pred['bboxes'])
        scores = np.array(pred['scores'])
        labels = np.array(pred['labels'])
    
        # plt.figure(figsize=(8, 8))
        # plt.imshow(result_roi)
        # plt.title(f"result_roi {img_info['file_name']}  |  ID: {img_id}")
        # plt.axis('off')
        # plt.savefig(os.path.join(output_path, f'result_roi_{MODEL_KEY}.jpg'))
    
        # plt.show()   
        
        plt.imsave(os.path.join(output_path, f'result_roi_{MODEL_KEY}.jpg'), result_roi)

        # Prepare cropper
        img_h, img_w = image.shape[:2]

       
        roi_info = cropper.extract_roi(bboxes, scores, labels, img_w, img_h, img_id, th, mode='training')

        if roi_info is None:
            # ROI rejected in training mode → skip image
            continue
        
        roi, cls, class_name, score = roi_info


        x1, y1, x2, y2 = map(int, roi)
        roi_coco_format = [x1, y1, x2 - x1, y2 - y1]
        
        # Update COCO image entry
        image_entry = image_dict[img_id]
        image_entry["roi_bbox"] = roi_coco_format
        image_entry["roi_class_id"] = cls
        image_entry["roi_class_name"] = class_name
        image_entry["roi_score"] = score
        
        
        # roi = [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, roi)
        
        # Crop ROI from the original image
        roi_img = image[y1:y2, x1:x2]

        # plt.figure(figsize=(8, 8))
        # plt.imshow(roi_img)
        # plt.title(f"result_roi {img_info['file_name']}  |  ID: {img_id}")
        # plt.axis('off')
        # plt.savefig(os.path.join(output_path, f'roi_img_{MODEL_KEY}.jpg'))
    
        # plt.show()  
        plt.imsave(os.path.join(output_path, f'cropped_roi_img_{MODEL_KEY}.jpg'), roi_img)

    
    output_ann_path = os.path.join(ann_dir_path, f"annotations_{dataset}{s_fixed_ann}_with_roi.json")
    

    with open(output_ann_path, "w") as f:
        json.dump(coco_json, f, indent=2)
    
    print(f"Saved updated COCO annotations with ROI --> {output_ann_path}")


    stats = cropper.get_statistics()
    cropper.show_hist(os.path.join('output', f'hist_th_{th}_{MODEL_KEY}.jpg'))

    print(f"\nSUMMARY STATS for model {MODEL_KEY}, th={th}, {dataset} dataset:")#, added boat, airplane, train classes:")
    for k, v in stats.items():
        print(f"{k:30s} : {v}")
    
    avg_time = np.mean(times) if times else 0
    print(f'Model {MODEL_KEY} took {avg_time:.03f} seconds in average over {len(subset_ids)} datapoints.')

    print('done')
    