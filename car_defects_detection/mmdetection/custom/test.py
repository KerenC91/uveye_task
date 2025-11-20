import os
import sys
import time
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from configs.config import test_params
from utils.data_utils import load_model, load_image_rgb
import argparse
import random

def draw_ground_truth(image, anns, coco):
    """Draw segmentation masks + bboxes for GT."""
    vis_img = image.copy()

    for ann in anns:
        color = tuple(np.random.randint(0, 255, 3).tolist())

        if ann.get("segmentation"):
            mask = coco.annToMask(ann)
            vis_img[mask == 1] = (
                vis_img[mask == 1] * 0.5 + np.array(color) * 0.5
            )

        x, y, w, h = ann["bbox"]
        cv2.rectangle(vis_img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        cat_name = coco.loadCats(ann["category_id"])[0]["name"]
        cv2.putText(vis_img, cat_name, (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis_img

def convert_predictions_to_coco(pred, img_id, cat_ids):
    """Convert MMDet predictions to COCO bbox format."""
    results = []

    for cls, score, bbox in zip(pred["labels"], pred["scores"], pred["bboxes"]):
        category_id = cat_ids[int(cls)]
        x1, y1, x2, y2 = bbox
        results.append({
            "image_id": img_id,
            "category_id": int(category_id),
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })

    return results


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    """Evaluate model on COCO annotations and compute metrics."""

    if not os.path.exists(args.img_dir):
        print("Error: dataset path doesn't exist.")
        sys.exit(1)

    if not os.path.exists(args.ann_path):
        print("Error: annotation file doesn't exist.")
        sys.exit(1)

    os.makedirs(args.output_folder, exist_ok=True)

    # ------------------------ Load COCO ------------------------
    print(f"Input annotation file: {args.ann_path}\n")
    coco = COCO(args.ann_path)

    image_dict = coco.imgs
    cat_ids = coco.getCatIds()

    # ------------------------ Load Model ------------------------
    inferencer, class_names = load_model(args.model_key, test_params.model_zoo)
    print(f"Using model: {args.model_key}")

    # ------------------------ Choose subset ---------------------
    img_ids = coco.getImgIds()
    subset_ids = (
        random.sample(img_ids, test_params.num_images)
        if test_params.num_images > 0
        else img_ids
    )
    print(f"Selected images: {'all' if test_params.num_images == 0 else len(subset_ids)}")

    times = []
    coco_results = []

    # ============================================================
    #                        MAIN LOOP
    # ============================================================
    for img_id in subset_ids:
        img_info = image_dict[img_id]
        img_path = os.path.join(args.img_dir, img_info["file_name"])

        image = load_image_rgb(img_path)
        if image is None:
            print(f"Warning: failed to load {img_path}")
            continue

        out_dir = os.path.join(args.output_folder, str(img_id))
        os.makedirs(out_dir, exist_ok=True)

        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))

        # ------------------------ GT VIS -------------------------
        vis_gt = draw_ground_truth(image, anns, coco)

        if args.save_debug_vis:
            cv2.imwrite(os.path.join(out_dir, "ground_truth.jpg"),
                        cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR))

        # ------------------------ PRED ---------------------------
        torch.cuda.synchronize()
        t0 = time.time()

        result = inferencer(image, return_vis=True)

        torch.cuda.synchronize()
        times.append(time.time() - t0)

        vis_pred = result["visualization"][0]

        if args.save_debug_vis:
            cv2.imwrite(os.path.join(out_dir, "prediction.jpg"),
                        cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR))

        pred = result["predictions"][0]
        coco_results.extend(convert_predictions_to_coco(pred, img_id, cat_ids))

    # --------------------- Timing ---------------------
    print(f"\nAverage inference time: {np.mean(times):.3f} sec over {len(subset_ids)} images")
    print("Running COCO evaluation...")

    # --------------------- COCO Evaluation ---------------------
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # --------------------- Per-Class AP ---------------------
    precisions = coco_eval.eval['precision']
    cats = coco.loadCats(cat_ids)

    print("\nPer-class AP (bbox):")
    for idx, cat in enumerate(cats):
        cls_prec = precisions[:, :, idx, 0, 2]
        cls_prec = cls_prec[cls_prec > -1]
        ap = cls_prec.mean() if cls_prec.size else float('nan')
        print(f"{cat['name']:15s}: {ap:.6f}")

    print("\nDone.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test car defects detection model')

    parser.add_argument('--img_dir', type=str, default='../data/Dataset/test2017')
    parser.add_argument('--ann_path', type=str, default='../data/Dataset/annotations/annotations_test_postprocessroi.json')
    parser.add_argument('--model_key', type=str, default='mask_rcnn_r50')
    parser.add_argument('--save_debug_vis', action='store_true')
    parser.add_argument('--output_folder', type=str, default="output/output_test/mask_rcnn_r50")

    args = parser.parse_args()
    main(args)
