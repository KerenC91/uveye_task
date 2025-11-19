import os
import sys
import json
import time
import random
import cv2
import torch
import numpy as np
#import matplotlib
#matplotlib.use("Agg")   # non-GUI backend for servers
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from mmdet.apis import DetInferencer
from configs.config import test_params
from utils.roi_cropper import ROICropper
from utils.general_utils import get_model_config
    
# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == '__main__':

    # Validate path
    if not os.path.exists(test_params.data_dir):
        print("Error: dataset path doesn't exist.")
        sys.exit(1)

    ann_dir = os.path.join(test_params.data_dir, "annotations")
    img_dir = os.path.join(test_params.data_dir, f"{test_params.dataset}2017")

    if not os.path.exists(ann_dir) or not os.path.exists(img_dir):
        print("Error: missing annotation or image folder.")
        sys.exit(1)

    # Ensure output root exists
    os.makedirs(test_params.output_root, exist_ok=True)
    # ============================================================
    #                   ANN PREPROCESSING
    # ============================================================
   
    ann_path = os.path.join(ann_dir, f"annotations_{test_params.dataset}_postprocess.json")

    with open(ann_path, "r") as f:
        coco = json.load(f)

    print(f"\nDataset: {test_params.dataset}")
    print(f"Input annotation file: {ann_path}\n")
    

    # ============================================================
    #                    LOAD DATASET + JSON
    # ============================================================
    coco = COCO(ann_path)

    with open(ann_path, "r") as f:
        coco_json = json.load(f)

    # Map image_id to image dict (for updating entries)
    image_dict = {img["id"]: img for img in coco_json["images"]}

    # ============================================================
    #                    LOAD MODEL + CROP LOGIC
    # ============================================================
    model_name, checkpoint = get_model_config(test_params.model_key, test_params.model_zoo)
    print(f"Using model: {test_params.model_key}")

    inferencer = DetInferencer(model_name, checkpoint)
    class_names = inferencer.model.dataset_meta["classes"]
    cropper = ROICropper(class_names)

    # Choose subset of image IDs
    img_ids = coco.getImgIds()
    subset_ids = (
        random.sample(img_ids, test_params.num_images)
        if test_params.num_images > 0
        else img_ids
    )

    print(f"Selected images: {'all' if test_params.num_images == 0 else len(subset_ids)}")

    times = []

    # ============================================================
    #                        MAIN LOOP
    # ============================================================
    for img_id in subset_ids:

        # ------------------ Load image --------------------------
        img_info = image_dict[img_id]
        img_path = os.path.join(img_dir, img_info["file_name"])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if image is None:
            print(f"Warning: failed to load {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        # ------------------ Output folder ------------------------
        out_dir = os.path.join(test_params.output_root, str(img_id))
        os.makedirs(out_dir, exist_ok=True)

        # ------------------ Load anns -----------------------------
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # Optional debug mask visualization
        if test_params.save_debug_vis:
            vis_img = image.copy()
        
            for ann in anns:
                color = tuple(np.random.randint(0, 255, 3).tolist())
        
                # blend segmentation mask
                if ann.get("segmentation"):
                    mask = coco.annToMask(ann)
                    vis_img[mask == 1] = (
                        vis_img[mask == 1] * 0.5 + np.array(color) * 0.5
                    )
        
                # draw bbox
                x, y, w, h = ann["bbox"]
                cv2.rectangle(vis_img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                cat_name = coco.loadCats(ann["category_id"])[0]["name"]
                cv2.putText(
                    vis_img,
                    cat_name,
                    (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
        
            plt.imsave(os.path.join(out_dir, "ground_truth.jpg"), vis_img)

        # ============================================================
        #                       DETECTOR RUN
        # ============================================================
        torch.cuda.synchronize()
        start = time.time()

        result = inferencer(image, return_vis=True)

        torch.cuda.synchronize()
        times.append(time.time() - start)

        vis_pred = result["visualization"][0]
        pred = result["predictions"][0]

        # Save visualization
        if test_params.save_debug_vis:
            plt.imsave(
                os.path.join(out_dir, "prediction.jpg"),
                vis_pred
            )

    print('done')
