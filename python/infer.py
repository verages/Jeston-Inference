import cv2
import numpy as np
import random
from typing import List, Tuple, Union
from utils.pycuda_api import TRTEngine


CLASSES_DET = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES_DET)
}

class YoloEngineInfer:
    def __init__(self, engine_path, confidence):
        self.engine = TRTEngine(engine_path)
        self.confidence = confidence
        self.h, self.w = 640, 640
        self.infer_num = 0


    def letterbox(self,im: np.ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[np.ndarray, float, Tuple[float, float]]:
        shape = im.shape[:2] 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
            1] 
        dw /= 2 
        dh /= 2
        if shape[::-1] != new_unpad: 
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im,
                                top,
                                bottom,
                                left,
                                right,
                                cv2.BORDER_CONSTANT,
                                value=color) 
        return im, r, (dw, dh)
   
    def blob(self, im: np.ndarray, return_seg: bool = False) -> Union[np.ndarray, Tuple]:
        seg = None
        if return_seg:
            seg = im.astype(np.float32) / 255
        im = im.transpose([2, 0, 1])
        im = im[np.newaxis, ...]
        im = np.ascontiguousarray(im).astype(np.float32) / 255
        if return_seg:
            return im, seg
        else:
            return im 
    def preprocess(self, img):
        bgr = img.copy()
        bgr, ratio, dwdh = self.letterbox(bgr, (self.w, self.h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)

        return img, dwdh, ratio, tensor

    def infer(self, img, infer_status):
        if infer_status:
            draw, dwdh, ratio, tensor = self.preprocess(img)
            data = self.engine(tensor)
            bboxes, scores, labels = self.postprocess(data)
            if bboxes.size == 0:
                print('no object!')
            else:
                save_path = f'./result/{self.infer_num}.jpg'
                self.infer_num += 1
                print(f'find {len(bboxes)} objects')
                bboxes -= dwdh
                bboxes /= ratio
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    
                    bbox = bbox.round().astype(np.int32).tolist()
                    cls_id = int(label)
                    cls = CLASSES_DET[cls_id]
                    color = COLORS[cls]

                    text = f'{cls}:{score:.3f}'
                    x1, y1, x2, y2 = bbox

                    (_w, _h), _bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.8, 1)
                    _y1 = min(y1 + 1, draw.shape[0])

                    cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(draw, (x1, _y1), (x1 + _w, _y1 + _h + _bl),
                                (0, 0, 255), -1)
                    cv2.putText(draw, text, (x1, _y1 + _h), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 255, 255), 2)
                cv2.imwrite(save_path, draw)
    

    def postprocess(self,data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        assert len(data) == 4
        num_dets, bboxes, scores, labels = (i[0] for i in data)
        nums = num_dets.item()
        if nums == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty(
                (0, ), dtype=np.float32), np.empty((0, ), dtype=np.int32)
        # check score negative
        scores[scores < 0] = 1 + scores[scores < 0]
        bboxes = bboxes[:nums]
        scores = scores[:nums]
        labels = labels[:nums]
        return bboxes, scores, labels

if __name__ == '__main__':
    YoloInfer = YoloEngineInfer('yolo11n.engine', 0.25)
    img = cv2.imread('test.jpg')
    YoloInfer.infer(img, True)
    