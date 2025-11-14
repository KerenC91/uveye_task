import numpy as np
from collections import defaultdict

class ROICropper:
    def __init__(self, class_names=None):
        # Statistics
        self.class_counter = defaultdict(int)
        self.class_scores_sum = defaultdict(float)
        self.global_scores = []
        self.class_names = class_names

    def extract_roi(self, bboxes, scores, labels, img_w, img_h):
        """
        bboxes: Nx4 array [x1, y1, x2, y2]
        scores: Nx1 array
        labels: Nx1 array
        img_w, img_h: fallback ROI when no detection
        """

        # No predictions → full-image ROI
        if bboxes is None or len(bboxes) == 0:
            return [0, 0, img_w, img_h], None, 0.0

        # Compute areas
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        idx = np.argmax(areas)

        roi = bboxes[idx].tolist()
        cls = int(labels[idx])
        score = float(scores[idx])
        class_name = self.class_names[cls] if self.class_names else None
        
        # Update stats
        self.class_counter[cls] += 1
        self.class_scores_sum[cls] += score
        self.global_scores.append(score)

        return roi, cls, class_name, score

    def get_statistics(self):
        # Per class accuracy
        per_class_stats = {}
        for cls in self.class_counter:
            count = self.class_counter[cls]
            avg_score = self.class_scores_sum[cls] / count
            per_class_stats[cls] = {
                "count": count,
                "avg_accuracy": avg_score
            }

        # Global metrics
        global_avg = float(np.mean(self.global_scores)) if self.global_scores else 0.0
        global_min = float(np.min(self.global_scores)) if self.global_scores else 0.0

        return {
            "per_class": per_class_stats,
            "global_avg_accuracy": global_avg,
            "global_min_accuracy": global_min
        }
