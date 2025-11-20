import os
import sys
import json
import time
import cv2
import torch
import numpy as np
from configs.config import params
from utils.annotations_vlaidation import preprocess_annotations
from utils.roi_cropper import ROICropper
from utils.general_utils import make_file_logger
from utils.data_utils import select_image_ids, load_model, load_image_rgb, extract_valid_category_ids
import argparse


# --------------------------------------------------------------------
# Annotation Preprocessing
# --------------------------------------------------------------------
def load_and_clean_annotations(path, output_folder):
    """
    Load COCO annotations and run preprocessing/cleanup.
    """
    with open(path, "r") as f:
        coco = json.load(f)

    valid_ids = extract_valid_category_ids(coco)
    log_path = os.path.join(output_folder, "logs/preprocess_annotations.log")
    ann_logger = make_file_logger(log_path)

    cleaned_anns, stats = preprocess_annotations(coco, valid_ids, logger=ann_logger)

    log_ann_stats(stats, ann_logger)
    
    coco_fixed = {
        "licenses": coco["licenses"],
        "info": coco["info"],
        "images": coco["images"],
        "annotations": cleaned_anns,
        "categories": coco["categories"],
    }

    return coco_fixed, stats, log_path

def log_ann_stats(stats, logger):
    logger("\nSUMMARY ANN STATS:")
    for k, v in stats.items():
        logger(f"{k:30s}: {v}")

def log_stats(stat_dict, logger):
    """Log ROI statistics in a structured format."""
    global_avg = stat_dict["global_avg_accuracy"]
    global_min = stat_dict["global_min_accuracy"]

    logger("=== ROI Extraction Statistics ===")

    logger(f"Global average score : {global_avg:.4f}")
    logger(f"Global minimum score : {global_min:.4f}")
    logger("")

    # Core stats
    logger("Rejection / Acceptance Summary:")
    for key, entry in stat_dict["stats"].items():
        logger(f"{key}:")
        logger(f"  count = {entry['count']}")
        logger(f"  ids   = {entry['ids']}")

    logger("")

    # Class-level stats
    logger("Per-Class Summary:")
    for cname, info in stat_dict["class_stats"].items():
        logger(f"  {cname:20s} count={info['count']:<5d}  avg_score={info['avg_accuracy']:.4f}")

    logger("")

    # Score buckets
    logger("Score Buckets (img_id, score):")
    for bucket_name, items in stat_dict["score_buckets"].items():
        logger(f"  {bucket_name:5s}: {len(items)} samples")

# --------------------------------------------------------------------
# Single Image ROI Extraction Step
# --------------------------------------------------------------------
def process_single_image(
    inferencer,
    cropper,
    image_path,
    img_id,
    image_dict,
    output_folder,
    save_debug,
):
    """
    Run model on a single image and extract ROI if available.
    """  
    image = load_image_rgb(image_path)
    if image is None:
        print(f"Warning: failed to load {image_path}")
        return None, None
    
    h, w = image.shape[:2]
    out_dir = os.path.join(output_folder, str(img_id))
    if save_debug:
        os.makedirs(out_dir, exist_ok=True)

    torch.cuda.synchronize()
    start = time.time()
    result = inferencer(image, return_vis=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    vis_pred = result["visualization"][0]
    pred = result["predictions"][0]

    if save_debug:
        cv2.imwrite(
            os.path.join(out_dir, "roi_detection.jpg"),
            cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR),
        )

    roi_info = cropper.extract_roi(
        np.array(pred["bboxes"]),
        np.array(pred["scores"]),
        np.array(pred["labels"]),
        w,
        h,
        img_id,
        mode=params.mode,
    )

    if roi_info is None:
        return elapsed, None

    roi, cls_id, cls_name, score = roi_info
    x1, y1, x2, y2 = map(int, roi)
    roi_coco = [x1, y1, x2 - x1, y2 - y1]

    img_entry = image_dict[img_id]
    img_entry["roi_bbox"] = roi_coco

    if save_debug:
        cropped = image[y1:y2, x1:x2]
        cv2.imwrite(
            os.path.join(out_dir, "cropped.jpg"),
            cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR),
        )

    return elapsed, img_entry


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(args):
    """
    Execute full preprocessing pipeline: clean annotations, detect cars, crop ROIs, write output.
    """
    if not os.path.exists(args.img_dir):
        print("Error: dataset path doesn't exist.")
        sys.exit(1)

    os.makedirs(args.output_folder, exist_ok=True)

    # ---------------------- ANN PREPROCESSING ----------------------
    coco, ann_stats, ann_log_path = load_and_clean_annotations(
        args.in_ann_path, args.output_folder
    )

    print(f"Annotation logs saved to {ann_log_path}")

    # Build index
    image_dict = {img["id"]: img for img in coco["images"]}
    subset_ids = select_image_ids(coco["images"], params.num_images)

    print(f"Selected images: {'all' if params.num_images == 0 else len(subset_ids)}")

    # ----------------------- Load model ----------------------------
    inferencer, class_names = load_model(args.model_key, params.model_zoo)
    cropper = ROICropper(class_names)

    # ----------------------- Main Loop -----------------------------
    times = []
    post_images = []

    for img_id in subset_ids:
        img_info = image_dict[img_id]
        img_path = os.path.join(args.img_dir, img_info["file_name"])

        elapsed, img_entry = process_single_image(
            inferencer,
            cropper,
            img_path,
            img_id,
            image_dict,
            args.output_folder,
            args.save_debug_vis,
        )

        if elapsed is not None:
            times.append(elapsed)
        if img_entry is not None:
            post_images.append(img_entry)

    coco["images"] = post_images

    # ---------------------- Save Output ----------------------------
    print(args.out_ann_path)
    os.makedirs(os.path.dirname(args.out_ann_path), exist_ok=True)

    with open(args.out_ann_path, "w") as f:
        json.dump(coco, f, indent=2)

    print("\nSaved updated COCO annotations with ROI to")
    print(args.out_ann_path)

    # ---------------------- ROI Stats ------------------------------
    roi_stats = cropper.get_statistics()

    if args.save_debug_vis:
        hist_path = os.path.join(
            args.output_folder, f"hist_th_{params.score_threshold}_{args.model_key}.jpg"
        )
        cropper.show_hist(hist_path)

    print(f"\nAverage inference time: {np.mean(times):.3f} sec")

    roi_log_path = os.path.join(args.output_folder, "logs/preprocess_roi.log")
    roi_logger = make_file_logger(roi_log_path)

    log_stats(roi_stats, roi_logger)

    print(f"ROI stats saved to {roi_log_path}")
    print("Done.")

        

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