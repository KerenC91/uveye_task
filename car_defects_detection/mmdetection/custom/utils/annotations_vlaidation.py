from collections import defaultdict
import numpy as np


def clip_box_to_image(box, W, H):
    """
    Clip COCO bbox [x, y, w, h] to image boundaries [0..W-1], [0..H-1].
    Returns:
        new_box or None if invalid after clipping.
    """
    if len(box) != 4:
        return None

    x, y, w, h = box
    x2, y2 = x + w, y + h

    # Fix reversed width or height (if annotator accidentally swapped)
    if w < 0:
        x, x2 = x2, x
    if h < 0:
        y, y2 = y2, y

    # Clip coordinates
    x  = max(0, min(x,  W - 1))
    y  = max(0, min(y,  H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))

    if x2 <= x or y2 <= y:
        return None

    return [float(x), float(y), float(x2 - x), float(y2 - y)]

def clip_polygon(poly, xend, yend, xstart=0, ystart=0):
    """
    Clip polygon segmentation list [x1, y1, x2, y2, ...] to image bounds.
    Minimum required length is 6 (3 points).
    """
    if len(poly) < 6:
        return None

    clipped = []
    for i in range(0, len(poly), 2):
        x = float(np.clip(poly[i],     xstart, xend))
        y = float(np.clip(poly[i + 1], ystart, yend))
        clipped.extend([x, y])

    return clipped if len(clipped) >= 6 else None

def fix_category_id(cid, valid_ids, ann_id, stats, logger):
    """
    Validate or reject category_id.
    Returns:
        fixed ID or None: drop annotation.
    """
    if cid is None:
        stats["missing_category_id"] += 1
        logger(f"[SKIP] ann {ann_id}: missing category_id")
        return None

    if cid not in valid_ids:
        stats["invalid_category_ids"] += 1
        logger(f"[DROP] ann {ann_id}: invalid category_id={cid}")
        return None

    return cid

def box_iou(b1, b2):
    """
    Compute IoU for COCO boxes [x, y, w, h].
    """
    x1, y1, w1, h1 = b1
    x2, y2 = x1 + w1, y1 + h1

    xx1, yy1, ww1, hh1 = b2
    xx2, yy2 = xx1 + ww1, yy1 + hh1

    inter_x1 = max(x1,  xx1)
    inter_y1 = max(y1,  yy1)
    inter_x2 = min(x2,  xx2)
    inter_y2 = min(y2,  yy2)

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
    """
    Remove duplicated annotations with matching category_id and IoU > thresh.
    Deterministic order via sorting by annotation ID.
    """
    anns = sorted(anns, key=lambda a: a["id"])
    keep = []

    for ann in anns:
        is_dup = False
        for kept in keep:
            if ann["category_id"] == kept["category_id"]:
                if box_iou(ann["bbox"], kept["bbox"]) > thresh:
                    is_dup = True
                    break

        if not is_dup:
            keep.append(ann)

    return keep

def annotation_is_consistent(ann, image_dict, stats, logger):
    """
    Validate minimal required fields & references.
    """
    ann_id = ann.get("id")

    # Image ref required
    if "image_id" not in ann:
        stats["missing_image_id"] += 1
        logger(f"[SKIP] ann {ann_id}: missing image_id")
        return False

    if ann["image_id"] not in image_dict:
        stats["ann_ref_missing_image"] += 1
        logger(f"[SKIP] ann {ann_id}: references missing image_id={ann['image_id']}")
        return False

    # Bbox required
    if "bbox" not in ann:
        stats["missing_bbox"] += 1
        logger(f"[SKIP] ann {ann_id}: missing bbox")
        return False

    # Segmentation optional, but must be list format
    if "segmentation" in ann:
        if not isinstance(ann["segmentation"], list):
            stats["segmentation_invalid_format"] += 1
            logger(f"[SKIP] ann {ann_id}: invalid segmentation format")
            return False

    return True

def heuristic_fix_annotation(ann):
    """
    Add missing fields & normalize segmentation.
    """
    if "iscrowd" not in ann:
        ann["iscrowd"] = 0

    if ann.get("segmentation") is None:
        ann["segmentation"] = []

    return ann


# ------------------------------------------------------------ #
# Main
# ------------------------------------------------------------ #
def preprocess_annotations(coco, valid_ids, logger=print):
    """
    Full COCO preprocessing pipeline:
      - consistency checks
      - category fix
      - bbox + polygon clipping
      - heuristic fixes
      - deduplication
    """

    stats = defaultdict(int)
    image_dict = {img["id"]: img for img in coco["images"]}
    cleaned = []

    for ann in coco["annotations"]:
        ann_id = ann.get("id")

        # structural consistency
        if not annotation_is_consistent(ann, image_dict, stats, logger):
            continue

        img = image_dict[ann["image_id"]]
        W, H = img["width"], img["height"]

        # category validation
        cid = fix_category_id(ann.get("category_id"), valid_ids, ann_id, stats, logger)
        if cid is None:
            continue

        ann["category_id"] = cid

        # heuristic cleanup
        ann = heuristic_fix_annotation(ann)

        # clip bbox
        fixed_bbox = clip_box_to_image(ann["bbox"], W, H)
        if fixed_bbox is None:
            stats["invalid_bbox"] += 1
            logger(f"[SKIP] ann {ann_id}: invalid bbox after clipping")
            continue

        ann["bbox"] = fixed_bbox

        # fix segmentation polygons
        new_segs = []
        for poly in ann.get("segmentation", []):
            clipped_poly = clip_polygon(poly, W - 1, H - 1)
            if clipped_poly:
                new_segs.append(clipped_poly)

        ann["segmentation"] = new_segs

        if len(new_segs) == 0:
            stats["segmentation_empty"] += 1
            logger(f"[SKIP] ann {ann_id}: segmentation empty after fixing")
            continue

        cleaned.append(ann)
        stats["kept"] += 1

    # deduplicate final annotations
    logger("Running deduplication...")
    cleaned = deduplicate_annotations(cleaned, thresh=0.9)

    stats["final_count"] = len(cleaned)
    return cleaned, dict(stats)


