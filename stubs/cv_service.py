from typing import List, Any
from tilsdk.cv.types import *
import onnxruntime as ort
import os
import cv2
import numpy as np
from torchvision import transforms
import torch

class CVService:
    def __init__(self, model_dir):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''

        # TODO: Participant to complete.
        print('using ONNX for CV')
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, 'frcnn.onnx')  # TODO: try other models
        self.session =  ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider',  'CPUExecutionProvider',])
        self.det_threshold = 0.8  # TODO: decide on good det_threshold
        self.input_size = 720 # 800
        self.id = 0

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def targets_from_image(self, img: np.array) -> List[DetectedObject]:
        '''Process image and return targets.

        Parameters
        ----------
        img : Any
            Input image.

        Returns
        -------
        results  : List[DetectedObject]
            Detected targets. 1 for fallen, 0 for standing

        Impt: DetectedObject postprocessing logic
        1 pt TP/TN
        0.5 pt Bbox correct but wrong cls
        -0.5 pt wrong bbox

        Idea: Choose between top x bboxes or bboxes above certain thresh (tricky part: threshold for robot captured images probably lower due to blur, so should try not to use hard-coded thresholds)

        Approach: Above threshold=0.8 , take top 0-3 images
        If no candidates, no detection.
        If 1 candidate, 1 detection.
        If difference in score for top1 and top2 > 0.1, just return top 1 (rationale: top1 probably human instance)
        Else return largest bbox area
        '''

        # res = []
        candidates = []
        h, w = img.shape[0], img.shape[1]
        img_tensor = transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize((720, 720)),
                                 transforms.ToTensor()
        ])(img)
        img_np = self.to_numpy(img_tensor)

        # img = img.astype(np.float32)
        # img = cv2.resize(img, dsize=(self.input_size, self.input_size),interpolation=cv2.INTER_AREA)
        # img = np.transpose(img, (2, 0, 1))
        # img = img / 255

        ort_inputs = {self.session.get_inputs()[0].name: img_np}
        ort_outs = self.session.run(None, ort_inputs)
        for i in range(len(ort_outs[0])):
            x1, y1, x2, y2 = ort_outs[0][i]
            x1 *= (w/self.input_size)
            x2 *= (w/self.input_size)
            y1 *= (h/self.input_size)
            y2 *= (h/self.input_size)

            # frcnn
            label = int(ort_outs[1][i])
            if label == 2:
                label = 0
            score = float(ort_outs[2][i])
            width = int(x2 - x1)
            height = int(y2 - y1)
            x = int((x1+x2) / 2)
            y = int((y1+y2) / 2)

            area = width*height
            bbox = BoundingBox(x, y, width, height)  # x_center, y_center, w, h
            candidates.append((score, area, bbox, label))

        target = self.postprocess_candidates(candidates)
        self.id += 1
        if not target:
            return []
        else:
            return [target]

        # postprocess detections
    def postprocess_candidates(self, candidates: List):
        candidates = [i for i in candidates if i[0] > self.det_threshold]
        if not candidates:
            return None

        if len(candidates) == 1:
            print('one candidate, score: ', candidates[0][0])
            return DetectedObject(self.id, candidates[0][3], candidates[0][2])

        candidates = sorted(candidates, reverse=True, key=lambda x:x[0])

        # APPROACH 1: using highest score
        # chosen = candidates[0]

        # APPROACH 2: using candidate rule
        top_candidates = candidates[:min(3, len(candidates))]

        # if top candidate is more than 0.1 score higher than the second top, return top
        if top_candidates[0][0] - top_candidates[0][1] > 0.1:
            print('top candidate score much better')
            chosen = top_candidates[0]
        else:
            print('taking largest area')
            top_candidates = sorted(top_candidates, reverse=True, key=lambda x:x[1])
            chosen = top_candidates[0]

        print('score', chosen[0])

        return DetectedObject(str(self.id), chosen[3], chosen[2])




class MockCVService:
    '''Mock CV Service.

    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        # Does nothing.
        pass

    def targets_from_image(self, img:Any) -> List[DetectedObject]:
        '''Process image and return targets.

        Parameters
        ----------
        img : Any
            Input image.

        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        # dummy data
        bbox = BoundingBox(100,100,300,50)
        obj = DetectedObject("1", "1", bbox)
        return [obj]
