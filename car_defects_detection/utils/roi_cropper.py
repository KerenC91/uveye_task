import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from configs.config import params


class ROICropper:
    """
    Car ROI extractor that:
      - filters detections by class
      - filters by score threshold
      - selects the largest-area box
      - clips bbox to image bounds
      - validates ROI size and area ratio
      - records statistics
    """

    def __init__(self, class_names=None):
        self.class_names = class_names

        # Global statistics
        self.class_counter = defaultdict(int)
        self.class_scores_sum = defaultdict(float)
        self.global_scores = []

        # Tracking various rejection cases
        self.full_area_counter = 0
        self.rejected_low_score = 0
        self.rejected_area_ratio = 0
        self.rejected_zero_size = 0
        self.accepted_counter = 0
        self.rejected_out_of_boundaries_counter = 0

        # Image ID tracking
        self.full_area_ids = []
        self.rejected_low_score_ids = []
        self.rejected_area_ratio_ids = []
        self.rejected_zero_size_ids = []
        self.rejected_out_of_boundaries_ids = []
        self.accepted_ids = []
        self.class_ids = defaultdict(list)
        
        # Score buckets: store (img_id, score)
        self.score_bucket_0_1   = []   # score < 0.1
        self.score_bucket_1_2   = []   # 0.1 ≤ score < 0.2
        self.score_bucket_2_3   = []   # 0.2 ≤ score < 0.3
        self.score_bucket_3_4   = []   # 0.3 ≤ score < 0.4
        self.score_bucket_4_5   = []   # 0.4 ≤ score < 0.5


    def _update_score_buckets(self, img_id, score):
        """Store image ID and score according to predefined buckets."""
        if score < 0.1:
            self.score_bucket_0_1.append((img_id, score))
        elif score < 0.2:
            self.score_bucket_1_2.append((img_id, score))
        elif score < 0.3:
            self.score_bucket_2_3.append((img_id, score))
        elif score < 0.4:
            self.score_bucket_3_4.append((img_id, score))
        elif score < 0.5:
            self.score_bucket_4_5.append((img_id, score))
        # scores ≥ 0.5 are handled normally by your threshold logic

    # ----------------------------------------------------------------------
    # MAIN ROI EXTRACTION
    # ----------------------------------------------------------------------
    def extract_roi(
        self,
        ann_bbox, bboxes, scores, labels,
        img_w, img_h, img_id,
        mode="training"
    ):
        """
        bboxes: Nx4 array float [x1, y1, x2, y2]
        scores: N
        labels: N
        mode: training/inference
        """

        # -----------------------------------------------------
        # 1. No predictions at all
        # -----------------------------------------------------
        if bboxes is None or len(bboxes) == 0:
            self.full_area_counter += 1
            self.full_area_ids.append(img_id)
            return None if mode == "training" else ([0, 0, img_w, img_h], None, None, 0.0)

        # -----------------------------------------------------
        # 2. Filter by allowed vehicle classes
        # -----------------------------------------------------
        idx = [i for i, lbl in enumerate(labels) if lbl in params.vehicle_classes]

        if len(idx) == 0:
            self.full_area_counter += 1
            self.full_area_ids.append(img_id)
            return None if mode == "training" else ([0, 0, img_w, img_h], None, None, 0.0)

        bboxes = bboxes[idx]
        scores = scores[idx]
        labels = labels[idx]

        # -----------------------------------------------------
        # 3. Score threshold
        # -----------------------------------------------------
        mask = scores >= params.score_threshold
        if not np.any(mask):
            self.rejected_low_score += 1
            self.rejected_low_score_ids.append(img_id)
            return None if mode == "training" else ([0, 0, img_w, img_h], None, None, 0.0)

        b = bboxes[mask].copy()
        sc = scores[mask]
        lb = labels[mask]

        # -----------------------------------------------------
        # 4. Compute areas after clipping
        # -----------------------------------------------------
        # clip: valid pixel coords = [0 .. w-1] / [0 .. h-1]
        b[:, 0] = np.clip(b[:, 0], 0, img_w - 1)   # x1
        b[:, 2] = np.clip(b[:, 2], 0, img_w - 1)   # x2
        b[:, 1] = np.clip(b[:, 1], 0, img_h - 1)   # y1
        b[:, 3] = np.clip(b[:, 3], 0, img_h - 1)   # y2

        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        idx_sorted = np.argsort(-areas)    # desc

        largest_idx = idx_sorted[0]
        largest_area = float(areas[largest_idx])

        # -----------------------------------------------------
        # 5. Extract the largest ROI
        # -----------------------------------------------------
        x1, y1, x2, y2 = [int(v) for v in b[largest_idx]]

		# Clip x1,y1 to valid pixel indices (0 .. w-1 / 0 .. h-1)
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))

		# Clip x2,y2 to valid exclusive upper bound (0 .. w / 0 .. h)
        x2 = max(0, min(int(x2), img_w))
        y2 = max(0, min(int(y2), img_h))

        cls = int(lb[largest_idx])
        score = float(sc[largest_idx])
        class_name = self.class_names[cls] if self.class_names else None
        
        # Final safety check: ensure non-zero ROI
        if x2 <= x1 or y2 <= y1:
            self.rejected_zero_size += 1
            self.rejected_zero_size_ids.append(img_id)
            return None if mode == "training" else ([0, 0, img_w, img_h], None, None, score)
        # ======================================================
        # annotation must be inside predicted ROI
        # ======================================================
        if ann_bbox is not None:
            ax, ay, aw, ah = ann_bbox
            ann_x1 = int(ax)
            ann_y1 = int(ay)
            ann_x2 = int(ax + aw)
            ann_y2 = int(ay + ah)
        
            if not self.bbox_contains([x1, y1, x2, y2],
                                 [ann_x1, ann_y1, ann_x2, ann_y2]):
                # Annotation is outside predicted ROI: reject
                self.rejected_out_of_boundaries_counter += 1
                self.rejected_out_of_boundaries_ids.append(img_id)
        
                return None if mode == 'training' else (
                    [0,0,img_w,img_h], None, None, score
                )
        # -----------------------------------------------------
        # 6. Validate area ratio (min/max allowed)
        # -----------------------------------------------------
        image_area = img_w * img_h
        area_ratio = largest_area / image_area

        if not (params.min_area_ratio <= area_ratio <= params.max_area_ratio):
            self.rejected_area_ratio += 1
            self.rejected_area_ratio_ids.append(img_id)
            return None if mode == "training" else ([0, 0, img_w, img_h], None, None, score)

        # -----------------------------------------------------
        # 7. Accept ROI
        # -----------------------------------------------------
        roi = [x1, y1, x2, y2]

        self.class_counter[class_name] += 1
        self.class_scores_sum[class_name] += score
        self.global_scores.append(score)
        self.accepted_counter += 1
        self.accepted_ids.append(img_id)
        self.class_ids[class_name].append(img_id)

        # bucket by score
        self._update_score_buckets(img_id, score)
        
        return roi, cls, class_name, score
    
    def bbox_contains(self, big_roi, small_box):
        bx1, by1, bx2, by2 = big_roi
        sx1, sy1, sx2, sy2 = small_box
    
        return (
            sx1 >= bx1 and
            sy1 >= by1 and
            sx2 <= bx2 and
            sy2 <= by2
        )


    # ----------------------------------------------------------------------
    # STATISTICS
    # ----------------------------------------------------------------------
    def get_statistics(self):
        per_class_stats = {}

        for cname in self.class_counter:
            count = self.class_counter[cname]
            avg_score = self.class_scores_sum[cname] / count
            per_class_stats[cname] = {
                "count": count,
                "avg_accuracy": avg_score,
            }

        global_avg = float(np.mean(self.global_scores)) if self.global_scores else 0.0
        global_min = float(np.min(self.global_scores)) if self.global_scores else 0.0

        return {
            "per_class": per_class_stats,
            "global_avg_accuracy": global_avg,
            "global_min_accuracy": global_min,
            "full_area_counter": self.full_area_counter,
            "rejected_low_score": self.rejected_low_score,
            "accepted_counter": self.accepted_counter,
            "rejected_area_ratio": self.rejected_area_ratio,
            "rejected_zero_size": self.rejected_zero_size,
            "rejected_out_of_boundaries_counter": self.rejected_out_of_boundaries_counter,

            "full_area_ids": self.full_area_ids,
            "rejected_low_score_ids": self.rejected_low_score_ids,
            "rejected_area_ratio_ids": self.rejected_area_ratio_ids,
            "rejected_zero_size_ids": self.rejected_zero_size_ids,
            "self.class_ids": self.class_ids,
            "rejected_out_of_boundaries_ids": self.rejected_out_of_boundaries_ids,
            "accepted_ids": self.accepted_ids,
            
            "score_bucket_0_1": self.score_bucket_0_1,
            "score_bucket_1_2": self.score_bucket_1_2,
            "score_bucket_2_3": self.score_bucket_2_3,
            "score_bucket_3_4": self.score_bucket_3_4,
            "score_bucket_4_5": self.score_bucket_4_5
        }


    # ----------------------------------------------------------------------
    # HISTOGRAM PLOT
    # ----------------------------------------------------------------------
    def show_hist(self, filename):
        plt.hist(self.global_scores, bins=20)
        plt.title("Score distribution")
        plt.savefig(filename)
        plt.show()
