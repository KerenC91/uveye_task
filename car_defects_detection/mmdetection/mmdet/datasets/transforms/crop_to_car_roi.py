from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class CropToCarROI:
    """Crop image using img_info['roi_bbox'] stored in COCO format [x, y, w, h].
    GT bboxes/masks are assumed already pre-processed offline and should not be touched here.
    """
    def __call__(self, results):

        # img_info = results.get('img_info', {})
        roi = results.get('roi_bbox', None)

        if roi is None:
            # print("NO ROI NO ROI NO ROI NO ROI NO ROI NO ROI")
            # print(results)
            return results  # no ROI for this image
        # else:
            # print("YES ROI YES ROI YES ROI YES ROI YES ROI YES ROI")
            # print(roi)

        # COCO format: x, y, width, height
        x1, y1, w, h = map(int, roi)
        x2 = x1 + w
        y2 = y1 + h

        img = results['img']
        H, W = img.shape[:2]

        # safety clipping
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        # crop only the image
        results['img'] = img[y1:y2, x1:x2].copy()
        results['img_shape'] = results['img'].shape

        return results

