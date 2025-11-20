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
from utils.general_utils import get_model_config
import argparse

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    
    # Validate path
    if not os.path.exists(args.img_dir):
        print("Error: dataset path doesn't exist.")
        sys.exit(1)
    if not os.path.exists(args.ann_path):
        print("Error: annotaion file path doesn't exist.")
        sys.exit(1)
        
    # Ensure output root exists
    os.makedirs(args.output_folder, exist_ok=True)
    # ============================================================
    #                   ANN PREPROCESSING
    # ============================================================
   
    ann_path = args.ann_path
    
    with open(ann_path, "r") as f:
        coco = json.load(f)

    print(f"Input annotation file: {ann_path}\n")
    

    # ============================================================
    #                    LOAD DATASET + JSON
    # ============================================================
    coco = COCO(ann_path)

    image_dict = {
        img_id: coco.imgs[img_id]
        for img_id in coco.imgs
    }

    # ============================================================
    #                    LOAD MODEL + CROP LOGIC
    # ============================================================
    model_name, checkpoint = get_model_config(args.model_key, test_params.model_zoo)
    print(f"Using model: {args.model_key}")

    inferencer = DetInferencer(model_name, checkpoint)

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
        img_path = os.path.join(args.img_dir, img_info["file_name"])

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if image is None:
            print(f"Warning: failed to load {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        # ------------------ Output folder ------------------------
        out_dir = os.path.join(args.output_folder, str(img_id))
        os.makedirs(out_dir, exist_ok=True)

        # ------------------ Load anns -----------------------------
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # Mask visualization
        
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
        
        if args.save_debug_vis:
            # save image
            cv2.imwrite(
                os.path.join(out_dir, "ground_truth.jpg"),
                cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)  # convert from RGB to BGR
            )
        else:
            # show 
            plt.figure(figsize=(10, 6))
            plt.imshow(vis_img)
            plt.axis("off")
            plt.show()

        # ============================================================
        #                       DETECTOR RUN
        # ============================================================
        torch.cuda.synchronize()
        start = time.time()

        result = inferencer(image, return_vis=True)

        torch.cuda.synchronize()
        times.append(time.time() - start)

        vis_pred = result["visualization"][0]

        # save\show visualization

        if args.save_debug_vis:
            # save image
            cv2.imwrite(
                os.path.join(out_dir, "prediction.jpg"),
                cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR)  # convert from RGB to BGR
                )
        else:
            plt.figure(figsize=(10, 6))
            plt.imshow(vis_pred)
            plt.axis("off")
            plt.show()


    print(f"\nAverage inference time: {np.mean(times):.3f} sec over {len(subset_ids)} images")
    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test car defects detection model')

    parser.add_argument('--img_dir', type=str, default='../data/Dataset/test2017',
                        help='Full path to the image data folder.')
    parser.add_argument('--ann_path', type=str, default='../data/Dataset/annotations/annotations_test_postprocessroi.json',
                        help='Full path to the annotations file.')
    parser.add_argument('--model_key', type=str, default='mask_rcnn_r50',
                        help='Model to evaluate. One of mask_rcnn_r50, rtmdet_tiny.')
    parser.add_argument('--save_debug_vis', action='store_false', help='Saves data by default.')
    parser.add_argument('--output_folder', type=str, default="output/output_test/mask_rcnn_r50", 
                        help='Full path to output folder.')
    
    args = parser.parse_args()

    main(args)
    

