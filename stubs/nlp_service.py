from typing import Iterable, List
from tilsdk.localization.types import *
import onnxruntime as ort
import os
import torchaudio
import librosa
import soundfile as sf
from io import BytesIO
# import torch.nn.functional as F
# from transformers import Wav2Vec2FeatureExtractor
# from NLPModels import HubertForSpeechClassification

class NLPService:
    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''

        # TODO: Participant to complete.
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, 'checkpoint-2270')
        self.sampling_rate = 16000
        # self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_path)
        # self.model = HubertForSpeechClassification.from_pretrained(self.model_path)

    # def speech_file_to_array_fn(self, path, sampling_rate):
    #     speech_array, _sampling_rate = torchaudio.load(path)
    #     resampler = torchaudio.transforms.Resample(_sampling_rate)
    #     speech = resampler(speech_array).squeeze().numpy()
    #     return speech


    # def predict(self, path, sampling_rate):
    #     speech = self.speech_file_to_array_fn(path, sampling_rate)
    #     features = self.feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    #     input_values = features.input_values

    #     logits = self.model(input_values).logits

    #     scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    #     return scores

    def locations_from_clues(self, clues:Iterable[Clue]) -> List[RealLocation]:
        '''Process clues and get locations of interest.

        Parameters
        ----------
        clues
            Clues to process.
                clue_id: int
                location: RealLocation
                audio: bytes

        Returns
        -------
        lois
            Locations of interest.
        '''

        # TODO: Participant to complete.
        locations = []
        for clue in clues:
            locations.append(clue.location)

            waveform, sr = sf.read(BytesIO(clue.audio))
            print('in nlp service')

            # speech, sr = torchaudio.load(audio)
            # speech = speech[0].numpy().squeeze()
            # speech = librosa.resample(np.asarray(speech), sr, self.sampling_rate)
            # outputs = self.predict(audio, self.sampling_rate)
            # pred_class = outputs.argmax()

        return locations


class MockNLPService:
    '''Mock NLP Service.

    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir:str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        pass

    def locations_from_clues(self, clues:Iterable[Clue]) -> List[RealLocation]:
        '''Process clues and get locations of interest.

        Mock returns location of all clues.
        '''
        locations = [c.location for c in clues]

        return locations
