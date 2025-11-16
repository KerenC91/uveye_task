import json
import numpy as np
from collections import defaultdict
import os
from datetime import datetime


def make_file_logger(log_path):
    """
    Returns a logger(msg) function that appends messages to a log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def logger(msg):
        with open(log_path, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {msg}\n")

    return logger

# ============================================================
# Phase 1 — Validation utilities
# ============================================================

def clip_box_to_image(box, W, H):
    """Fix & clip COCO box [x,y,w,h]."""
    if len(box) != 4:
        return None

    x, y, w, h = box
    x2, y2 = x + w, y + h

    # Fix reversed width/height if annotator swapped x2<x or y2<y
    if w < 0:
        x, x2 = x2, x
    if h < 0:
        y, y2 = y2, y

    # Clip
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))

    if x2 <= x or y2 <= y:
        return None

    return [float(x), float(y), float(x2 - x), float(y2 - y)]


def clip_polygon(poly, W, H):
    """Clip poly points; poly = list of floats."""
    if len(poly) < 6:
        return None

    out = []
    for i in range(0, len(poly), 2):
        x = float(np.clip(poly[i], 0, W - 1))
        y = float(np.clip(poly[i + 1], 0, H - 1))
        out.extend([x, y])

    return out if len(out) >= 6 else None


# ============================================================
# Phase 2 — Class ID correction
# ============================================================

def fix_category_id(cid, valid_ids, ann_id, stats, logger):
    """Fix or drop inconsistent class IDs."""
    if cid is None:
        stats['missing_category_id'] += 1
        logger(f"[SKIP] ann {ann_id} missing category_id")
        return None

    if cid not in valid_ids:
        stats["invalid_category_ids"] += 1
        logger(f"[DROP] ann {ann_id}: invalid category_id={cid}")
        return None

    return cid


# ============================================================
# Phase 3 — Deduplication (NMS-style)
# ============================================================

def box_iou(b1, b2):
    """IoU for COCO boxes [x,y,w,h]."""
    x1, y1, w1, h1 = b1
    x2, y2 = x1 + w1, y1 + h1

    xx1, yy1, ww1, hh1 = b2
    xx2, yy2 = xx1 + ww1, yy1 + hh1

    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = ww1 * hh1
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def deduplicate_annotations(anns, thresh=0.9):
    """Remove duplicates based on IoU threshold."""
    keep = []
    anns = sorted(anns, key=lambda a: a["id"])  # deterministic order

    for ann in anns:
        duplicate = False
        for k in keep:
            if ann["category_id"] == k["category_id"]:
                if box_iou(ann["bbox"], k["bbox"]) > thresh:
                    duplicate = True
                    break
        if not duplicate:
            keep.append(ann)

    return keep


# ============================================================
# Phase 4 — Annotation consistency checks
# ============================================================

def annotation_is_consistent(ann, image_dict, stats, logger):
    """Check required fields & references."""
    ann_id = ann.get("id")

    if "image_id" not in ann:
        stats["missing_image_id"] += 1
        logger(f"[SKIP] ann {ann_id} has no image_id")
        return False

    if ann["image_id"] not in image_dict:
        stats["ann_ref_missing_image"] += 1
        logger(f"[SKIP] ann {ann_id} references missing image_id={ann['image_id']}")
        return False

    if "bbox" not in ann:
        stats["missing_bbox"] += 1
        logger(f"[SKIP] ann {ann_id} missing bbox")
        return False

    # Segmentation can be empty but must be list-of-lists
    if "segmentation" in ann:
        if not isinstance(ann["segmentation"], list):
            stats["segmentation_invalid_format"] += 1
            logger(f"[SKIP] ann {ann_id} has invalid segmentation format")
            return False

    return True


# ============================================================
# Phase 5 — Heuristic corrections
# ============================================================

def heuristic_fix_annotation(ann):
    """Add defaults like missing iscrowd; ensure structures."""
    if "iscrowd" not in ann:
        ann["iscrowd"] = 0

    if "segmentation" in ann and ann["segmentation"] is None:
        ann["segmentation"] = []

    return ann


# ============================================================
# Phase 7 — main pipeline with comments
# ============================================================

def preprocess_annotations(coco, valid_ids, logger=print):
    """
    Full COCO preprocessing pipeline implementing steps 1–7.
    """

    stats = defaultdict(int)

    # -----------------------------
    # Build image lookup table
    # -----------------------------
    image_dict = {img["id"]: img for img in coco["images"]}

    cleaned = []

    # ====================================================
    # MAIN LOOP OVER ANNOTATIONS
    # ====================================================
    for ann in coco["annotations"]:
        ann_id = ann.get("id")

        # --------------------------
        # PHASE 4: Consistency checks
        # --------------------------
        if not annotation_is_consistent(ann, image_dict, stats, logger):
            continue

        img = image_dict[ann["image_id"]]
        W, H = img["width"], img["height"]

        # --------------------------
        # PHASE 2: Class ID checking
        # --------------------------
        cid = fix_category_id(
            ann.get("category_id"), valid_ids, 
            ann_id, stats, logger
        )
        if cid is None:
            continue
        ann["category_id"] = cid

        # --------------------------
        # PHASE 5: Heuristic fixes
        # --------------------------
        ann = heuristic_fix_annotation(ann)

        # --------------------------
        # PHASE 1: Fix bounding box
        # --------------------------
        fixed_bbox = clip_box_to_image(ann["bbox"], W, H)
        if fixed_bbox is None:
            stats["invalid_bbox"] += 1
            logger(f"[SKIP] ann {ann_id} invalid bbox after clipping")
            continue
        ann["bbox"] = fixed_bbox

        # --------------------------
        # PHASE 1: Fix segmentation
        # --------------------------
        new_segs = []
        for poly in ann.get("segmentation", []):
            fixed = clip_polygon(poly, W, H)
            if fixed:
                new_segs.append(fixed)
        ann["segmentation"] = new_segs

        if len(new_segs) == 0:
            stats["segmentation_empty"] += 1
            logger(f"[SKIP] ann {ann_id} empty segmentation after fix")
            continue

        # Keep annotation
        cleaned.append(ann)
        stats["kept"] += 1

    # ====================================================
    # PHASE 3: Deduplication
    # ====================================================
    logger("Running deduplication...")
    cleaned = deduplicate_annotations(cleaned, thresh=0.9)

    stats["final_count"] = len(cleaned)

    return cleaned, dict(stats)

def extract_valid_category_ids(coco):
    """
    Extracts all category_id values from COCO JSON categories list.
    Returns a sorted list of integers.
    """
    if "categories" not in coco:
        raise ValueError("COCO JSON missing 'categories' field.")

    valid_ids = sorted([cat["id"] for cat in coco["categories"]])
    return valid_ids


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    ## config ##
    dataset = 'val' # train, test, val
    ############
    data_dir = '../data/Dataset'
    ann_path = os.path.join(data_dir, 'annotations')
    ann_file = f'annotations_{dataset}'
    full_ann_path = os.path.join(ann_path, f'{ann_file}.json')

    with open(full_ann_path, "r") as f:
        coco = json.load(f)

    valid_ids = extract_valid_category_ids(coco)

    logger = make_file_logger(f"../logs/preprocess_{ann_file}.log")
    
    cleaned_anns, stats = preprocess_annotations(
        coco,
        valid_ids,
        logger=logger
    )

    output_ann_path = os.path.join(ann_path, f"annotations_{dataset}_fixed.json")
    
    fixed_coco = {
        "licenses": coco["licenses"],    # UNCHANGED
        "info": coco["info"],            # UNCHANGED
        "images": coco["images"],        # UNCHANGED
        "annotations": cleaned_anns,     # CLEANED ANNOTATIONS
        "categories": coco["categories"] # UNCHANGED
    }
    
    with open(output_ann_path, "w") as f:
        json.dump(fixed_coco, f, indent=2)
    
    print(f"Saved updated COCO annotations to {output_ann_path}")


    print("\nSUMMARY STATS:")
    for k, v in stats.items():
        print(f"{k:30s} : {v}")
