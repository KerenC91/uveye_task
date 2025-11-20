import os
import sys
import json
import time
import random
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-GUI backend for servers
from mmdet.apis import DetInferencer
import yaml
from configs.config import params
from utils.roi_cropper import ROICropper
from utils.annotations_vlaidation import extract_valid_category_ids, make_file_logger, preprocess_annotations
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


    # Ensure output root exists
    os.makedirs(args.output_folder, exist_ok=True)
    # ============================================================
    #                   ANN PREPROCESSING
    # ============================================================
   
    pre_ann_path = args.in_ann_path

    with open(pre_ann_path, "r") as f:
        coco = json.load(f)

    print(f"Input annotation file: {pre_ann_path}\n")
    
    valid_ids = extract_valid_category_ids(coco)
    ann_log_path = f"{args.output_folder}/logs/preprocess_annotations.log"
    ann_logger = make_file_logger(ann_log_path)

    cleaned_anns, stats = preprocess_annotations(coco, valid_ids, logger=ann_logger)
    
    print(f'Annotations logs saved to {ann_log_path}.')
    
    # Build new COCO file
    fixed_coco = {
        "licenses": coco["licenses"],
        "info": coco["info"],
        "images": coco["images"],
        "annotations": cleaned_anns,
        "categories": coco["categories"],
    }

    print("\nSUMMARY ANN STATS:")#todo:add to logger
    for k, v in stats.items():
        print(f"{k:30s}: {v}")

    # ============================================================
    #                   ROI Extraction
    # ============================================================
    # New containers for postprocessed annotations and images
    post_images = []

    # Build fast index mappings for lookups
    image_dict = {img["id"]: img for img in fixed_coco["images"]}

    # Choose subset of image IDs
    img_ids = [img["id"] for img in fixed_coco["images"]]
    subset_ids = (
        random.sample(img_ids, params.num_images)
        if params.num_images > 0
        else img_ids
    )
    
    print(f"Selected images: {'all' if params.num_images == 0 else len(subset_ids)}")

    # ============================================================
    #                    LOAD MODEL + CROP LOGIC
    # ============================================================
    model_name, checkpoint = get_model_config(args.model_key, params.model_zoo)
    print(f"Using model: {args.model_key}")

    inferencer = DetInferencer(model_name, checkpoint)
    class_names = inferencer.model.dataset_meta["classes"]
    cropper = ROICropper(class_names)

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
        if args.save_debug_vis:
            os.makedirs(out_dir, exist_ok=True)


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
        if args.save_debug_vis:
            cv2.imwrite(
                os.path.join(out_dir, "roi_detection.jpg"),
                cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR)  # convert from RGB to BGR
            )

        bboxes = np.array(pred["bboxes"])
        scores = np.array(pred["scores"])
        labels = np.array(pred["labels"])

        # ============================================================
        #                   ROI EXTRACTION + UPDATE JSON
        # ============================================================
        roi_info = cropper.extract_roi(
            bboxes,
            scores,
            labels,
            img_w,
            img_h,
            img_id,
            mode="training",#todo:remove mode
        )

        if roi_info is None:
            continue  # skip ROI for this image

        roi, cls_id, cls_name, score = roi_info

        # convert ROI to COCO [x,y,w,h]
        x1, y1, x2, y2 = map(int, roi)
        roi_coco = [x1, y1, x2 - x1, y2 - y1]
        
        # save updated image entry
        img_entry = image_dict[img_id]
        img_entry["roi_bbox"] = roi_coco
        post_images.append(img_entry)

        # save cropped ROI image
        if args.save_debug_vis:
            roi_img = image[y1:y2, x1:x2]
            cv2.imwrite(
                os.path.join(out_dir, "cropped.jpg"),
                cv2.cvtColor(roi_img, cv2.COLOR_RGB2BGR)  # convert from RGB to BGR
            )

    # update fixed_coco
    fixed_coco["images"] = post_images

    
    # ============================================================
    #                SAVE UPDATED JSON WITH ROI
    # ============================================================
    out_post_ann_fullpath = args.out_ann_path

    with open(out_post_ann_fullpath, "w") as f:
        json.dump(fixed_coco, f, indent=2)

    print("\nSaved updated COCO annotations with ROI to")
    print(out_post_ann_fullpath)

    # ============================================================
    #                    SHOW ROI CROP STATS
    # ============================================================
    stats = cropper.get_statistics()
    cropper.show_hist(os.path.join(args.output_folder, f"hist_th_{params.score_threshold}_{args.model_key}.jpg"),
                      args.save_debug_vis)

    print(f"\nAverage inference time: {np.mean(times):.3f} sec over {len(subset_ids)} images")

    # ------------------------------------
    # Save stats to YAML file
    # ------------------------------------
    roi_stats_path = f"{args.output_folder}/logs/preprocess_roi.log"
   
    with open(roi_stats_path, 'w') as f:
        yaml.dump(stats, f, sort_keys=False, default_flow_style=False)
        
    print(f'ROI stats saved to {roi_stats_path}.')
    print("\nDone.")

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess annotations for car defects detection model')

    parser.add_argument('--img_dir', type=str, default='../data/Dataset/test2017',
                        help='Full path for the image data folder.')
    parser.add_argument('--in_ann_path', type=str, default='../data/Dataset/annotations/annotations_test.json',
                        help='Full path to the input annotations file.')
    parser.add_argument('--out_ann_path', type=str, default='../data/Dataset/annotations/annotations_test_postprocessroi.json',
                        help='Full path to the input annotations file.')
    parser.add_argument('--model_key', type=str, default='rtmdet_x',
                        help='Model for roi cropping. One of rtmdet_tiny, rtmdet_x, yolo_s, yolo_x.')
    parser.add_argument('--save_debug_vis', action='store_true', help='Does not save data by default.')
    parser.add_argument('--output_folder', type=str, default="output/output_preprocess", help='Full path to output folder')
    
    args = parser.parse_args()

    main(args)