import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from configs.config import params


class ROICropper:
    """ROI selector that filters detections, validates boxes, and tracks statistics."""

    def __init__(self, class_names=None):
        """Initialize stats containers and class mappings."""
        self.class_names = class_names

        self.stats = {
            "full_area":          {"count": 0, "ids": []},
            "low_score":          {"count": 0, "ids": []},
            "area_ratio":         {"count": 0, "ids": []},
            "zero_size":          {"count": 0, "ids": []},
            "out_of_bounds":      {"count": 0, "ids": []},
            "unallowed_classes":  {"count": 0, "ids": []},
            "accepted":           {"count": 0, "ids": []},
        }

        self.class_stats = {
            "counts": defaultdict(int),
            "sum_scores": defaultdict(float),
            "ids": defaultdict(list),
        }
        
        self.global_scores = []

        self.score_buckets = {
            "0_1": [], "1_2": [], "2_3": [], "3_4": [], "4_5": []
        }

    # ------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------ #
    def _reject(self, key, img_id, w, h, mode, score=0.0):
        """Record a rejection event and return fallback ROI."""
        self.stats[key]["count"] += 1
        self.stats[key]["ids"].append(img_id)
        if mode == "discard":
            return None
        return [0, 0, w, h], None, None, score

    def _allowed_vehicle_indices(self, labels):
        """Return indices of vehicle-class detections."""
        return [i for i, lbl in enumerate(labels) if lbl in params.vehicle_classes]

    def _forbidden_trigger(self, labels, scores):
        """Check if any forbidden class exceeds the rejection threshold."""
        bad = [i for i, lbl in enumerate(labels) if lbl in params.invalid_classes]
        if not bad:
            return False
        return np.any(scores[bad] >= params.bad_threshold)

    def _clip_boxes(self, b, w, h):
        """Clip bounding boxes to image boundaries."""
        b[:, 0] = np.clip(b[:, 0], 0, w - 1)
        b[:, 2] = np.clip(b[:, 2], 0, w - 1)
        b[:, 1] = np.clip(b[:, 1], 0, h - 1)
        b[:, 3] = np.clip(b[:, 3], 0, h - 1)
        return b

    def _choose_best(self, b, scores, w, h):
        """Pick the highest-ranked ROI by size–score combination."""
        areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        norm = areas / (w * h)
        rank = np.sqrt(norm) * scores
        idx = np.argmax(rank)
        return idx, norm[idx], scores[idx]

    def _clip_xyxy(self, x1, y1, x2, y2, w, h):
        """Clamp integer box coordinates inside the image."""
        return (
            max(0, min(x1, w - 1)),
            max(0, min(y1, h - 1)),
            max(0, min(x2, w - 1)),
            max(0, min(y2, h - 1)),
        )

    def _update_bucket(self, img_id, score):
        """Place a score into its corresponding bucket."""
        if score < 0.1: self.score_buckets["0_1"].append((img_id, score))
        elif score < 0.2: self.score_buckets["1_2"].append((img_id, score))
        elif score < 0.3: self.score_buckets["2_3"].append((img_id, score))
        elif score < 0.4: self.score_buckets["3_4"].append((img_id, score))
        elif score < 0.5: self.score_buckets["4_5"].append((img_id, score))



    # ------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------ #
    def extract_roi(self, bboxes, scores, labels, img_w, img_h, img_id, mode="discard"):
        """Extract car ROI for an image and update statistics."""
        if bboxes is None or len(bboxes) == 0:
            return self._reject("full_area", img_id, img_w, img_h, mode)

        if self._forbidden_trigger(labels, scores):
            return self._reject("unallowed_classes", img_id, img_w, img_h, mode)

        idx = self._allowed_vehicle_indices(labels)
        if not idx:
            return self._reject("full_area", img_id, img_w, img_h, mode)

        b = bboxes[idx]
        sc = scores[idx]
        lb = labels[idx]

        mask = sc >= params.score_threshold
        if not np.any(mask):
            return self._reject("low_score", img_id, img_w, img_h, mode)

        b = b[mask].copy()
        sc = sc[mask]
        lb = lb[mask]

        b = self._clip_boxes(b, img_w, img_h)
        best_idx, area_ratio, best_score = self._choose_best(b, sc, img_w, img_h)

        x1, y1, x2, y2 = [int(v) for v in b[best_idx]]
        x1, y1, x2, y2 = self._clip_xyxy(x1, y1, x2, y2, img_w, img_h)

        if x2 <= x1 or y2 <= y1:
            return self._reject("zero_size", img_id, img_w, img_h, mode, best_score)

        if not (params.min_area_ratio <= area_ratio <= params.max_area_ratio):
            return self._reject("area_ratio", img_id, img_w, img_h, mode, best_score)

        cls = int(lb[best_idx])
        cname = self.class_names[cls] if self.class_names else None
        roi = [x1, y1, x2, y2]

        self.stats["accepted"]["count"] += 1
        self.stats["accepted"]["ids"].append(img_id)

        self.class_stats["counts"][cname] += 1
        self.class_stats["sum_scores"][cname] += best_score
        self.class_stats["ids"][cname].append(img_id)

        self.global_scores.append(best_score)
        self._update_bucket(img_id, best_score)

        return roi, cls, cname, best_score

    # ------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------ #
    def get_statistics(self):
        """Return accumulated statistics in a structured dict."""
        class_out = {
            cname: {
                "count": self.class_stats["counts"][cname],
                "avg_accuracy": (
                    self.class_stats["sum_scores"][cname] / self.class_stats["counts"][cname]
                ) if self.class_stats["counts"][cname] else 0.0
            }
            for cname in self.class_stats["counts"]
        }

        global_avg = float(np.mean(self.global_scores)) if self.global_scores else 0.0
        global_min = float(np.min(self.global_scores)) if self.global_scores else 0.0

        return {
            "stats": self.stats,
            "class_stats": class_out,
            "score_buckets": self.score_buckets,
            "global_avg_accuracy": global_avg,
            "global_min_accuracy": global_min,
        }

    # ------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------ #
    def show_hist(self, filename):
        """Save histogram of ROI acceptance scores."""
        plt.figure()
        plt.hist(self.global_scores, bins=20)
        plt.title("Score distribution")
        plt.savefig(filename)
        plt.close()
