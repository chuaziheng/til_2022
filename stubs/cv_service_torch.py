from typing import List, Any
from tilsdk.cv.types import *
import onnxruntime as ort
import os
import cv2
import numpy as np
import torchvision
from torchvision import transforms
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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
        print('using torch for CV')

        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, 'frcnn_torch_og.pt')
        self.det_threshold = 0.6  # # TODO: decide on good det_threshold
        self.input_size = 720 # 800
        self.id = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', self.device)
        # from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        # num_classes = 3

        # self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # ,trainable_backbone_layers=1
        # in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        # self.model.transform.min_size = (720,)
        # self.model.transform.max_size = 720
        # self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model = torch.load(self.model_path, map_location=torch.device(self.device))
        self.model.to(self.device)
        self.model.eval()


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
        # h, w = img.shape[0], img.shape[1]

        with torch.no_grad():
            img_tensor = transforms.ToTensor()(img)
            img_preds = self.model([img_tensor.to(self.device)])[0]

            for i in range(len(img_preds["boxes"])):
                x1, y1, x2, y2 = img_preds["boxes"][i]
                label = int(img_preds["labels"][i])
                score = float(img_preds["scores"][i])

                if label == 2:
                    label = 0
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

        # APPROACH 1: use highest score

        chosen = candidates[0]
        print('score', chosen[0])


        # APPROACH 2: use candidate rules

        # # get 0-3 candidates
        # top_candidates = candidates[:min(3, len(candidates))]

        # # if top candidate is more than 0.1 score higher than the second top, return top
        # if top_candidates[0][0] - top_candidates[0][1] > 0.1:
        #     print('top candidate score much better')
        #     chosen = top_candidates[0]
        # else:
        #     print('taking largest area')
        #     top_candidates = sorted(top_candidates, reverse=True, key=lambda x:x[1])
        #     chosen = top_candidates[0]
        return DetectedObject(self.id, chosen[3], chosen[2])




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
