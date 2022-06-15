from typing import List, Any
from tilsdk.cv.types import *
import onnxruntime as ort
from torchvision import transforms
import os

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


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
        self.det_threshold = 0.95

    def targets_from_image(self, img) -> List[DetectedObject]:
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

        # TODO: Participant to complete.
        res = []
        h, w = img.shape[0], img.shape[1]
        # assume that img is np.array
        img_tensor = transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize((800, 800)),
                                 transforms.ToTensor()
                                ])(img)
        img_np = to_numpy(img_tensor)
        ort_inputs = {self.session.get_inputs()[0].name: img_np}
        ort_outs = self.session.run(None, ort_inputs)
        for i in range(len(ort_outs[0])):
            x1, y1, x2, y2 = ort_outs[0][i]
            x1 *= (w/800)
            x2 *= (w/800)
            y1 *= (h/800)
            y2 *= (h/800)

            # frcnn
            label = int(ort_outs[1][i])
            score = float(ort_outs[2][i])
            width = int(x2 - x1)
            height = int(y2 - y1)

            # filter out non-confident detections
            print('score', score)
            if score > self.det_threshold:
                bbox = BoundingBox(x1, y1, width, height)
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
