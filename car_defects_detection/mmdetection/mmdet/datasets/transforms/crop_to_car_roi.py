from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from typing import Tuple, Union
import numpy as np

@TRANSFORMS.register_module()
class CropToCarROI(BaseTransform):
    """Crop image using img_info['roi_bbox'] stored in COCO format [x, y, w, h].
    GT bboxes/masks are assumed already pre-processed offline and should not be touched here.
    """
    
    def __init__(self,
                 allow_negative_crop: bool = False,
                 recompute_bbox: bool = False,
                 bbox_clip_border: bool = True) -> None:
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox
        
    #@autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        roi = results.get('roi_bbox', None)    
        # Must exist
        assert roi is not None, "roi_bbox missing in results"
        
        # Must be length 4
        assert len(roi) == 4, f"roi_bbox must have 4 values, got {len(roi)}"
        
        # All must be integers
        assert all(isinstance(v, (int, np.integer)) for v in roi), \
               f"roi_bbox must contain ints, got {roi}"
        
        # Must be non-negative
        assert all(v >= 0 for v in roi), \
               f"roi_bbox values must be >= 0, got {roi}"
                
        results = self._crop_data(results, roi)
        return results

    def _crop_data(self, results: dict, roi: Tuple[int, int, int, int]) -> Union[dict, None]:
        img = results['img']

        crop_y1, crop_y2 = roi[1], roi[1] + roi[3] 
        crop_x1, crop_x2 = roi[0], roi[0] + roi[2]
        
        assert crop_x2 >= crop_x1 and crop_y2 >= crop_y1

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -crop_x1], [0, 1, -crop_y1], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-crop_x1, -crop_y1])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not self.allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get('gt_instances_ids', None) is not None:
                results['gt_instances_ids'] = \
                    results['gt_instances_ids'][valid_inds]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results        
        

