from typing import List, Any
from tilsdk.cv.types import *
import onnxruntime as ort
import os
import cv2
import numpy as np



class CVService:
    def __init__(self, model_dir):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''

        # TODO: Participant to complete.
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, 'frcnn.onnx')
        self.session =  ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider',  'CPUExecutionProvider',])
        self.det_threshold = 0.8
        self.input_size = 800

    def targets_from_image(self, img: np.array) -> List[DetectedObject]:
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

        res = []
        h, w = img.shape[0], img.shape[1]
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(self.input_size, self.input_size),interpolation=cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
        img = img / 255

        ort_inputs = {self.session.get_inputs()[0].name: img}
        ort_outs = self.session.run(None, ort_inputs)
        for i in range(len(ort_outs[0])):
            x1, y1, x2, y2 = ort_outs[0][i]
            x1 *= (w/self.input_size)
            x2 *= (w/self.input_size)
            y1 *= (h/self.input_size)
            y2 *= (h/self.input_size)

            # frcnn
            label = int(ort_outs[1][i])
            score = float(ort_outs[2][i])
            width = int(x2 - x1)
            height = int(y2 - y1)
            x = int((x1+x2) / 2)
            y = int((y1+y2) / 2)
            # filter out non-confident detections
            print('score', score)
            if score > self.det_threshold:
                # bbox = BoundingBox(x1, y1, width, height)
                bbox = BoundingBox(x, y, width, height)  # x_center, y_center, w, h
                obj = DetectedObject("1", label, bbox)  # id ???
                res.append(obj)

        return res



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
