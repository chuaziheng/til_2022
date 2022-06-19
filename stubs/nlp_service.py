from typing import Iterable, List
from tilsdk.localization.types import *
import onnxruntime as ort
import os
import torchaudio
import librosa
import soundfile as sf
from io import BytesIO
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
        self.model_path = os.path.join(self.model_dir, 'speech.h5')
        self.model = tf.keras.models.load_model(self.model_path)
        self.CLASS_2_LABEL = {0: 'angry', 1: 'sad', 2: 'neutral', 3: 'happy', 4: 'fear'}
        # self.LABEL_2_CLASS = {'angry': 0, 'fear': 4, 'happy': 3, 'neutral': 2, 'sad': 1}

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
        def extract_features(data, sample_rate):
            # ZCR
            result = np.array([])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
            result=np.hstack((result, zcr)) # stacking horizontally
            # Chroma_stft
            stft = np.abs(librosa.stft(data))
            chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_stft)) # stacking horizontally
            # MFCC
            mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mfcc)) # stacking horizontally
            # Root Mean Square Value
            rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
            result = np.hstack((result, rms)) # stacking horizontally
            # MelSpectogram
            mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel)) # stacking horizontally
            return result

        # TODO: Participant to complete.
        locations = []
        print('in nlp service')

        for clue in clues:
            waveform, sr = sf.read(BytesIO(clue.audio))
            res1 = extract_features(waveform, sr)
            result = np.array(res1)
            pred_Features = pd.DataFrame(result)
            pred_X = pred_Features.values
            scaler = StandardScaler()
            pred_X = scaler.fit_transform(pred_X)
            pred_X = pred_X.T
            print(pred_X.shape)
            pred_test = self.model.predict(pred_X)
            y_pred = np.argmax(pred_test, axis=1)
            pred_lab = [self.CLASS_2_LABEL[i] for i in y_pred][0]
            print('predicted', pred_lab)
            if pred_lab in ['sad', 'angry', 'happy']:
                print('correct pred')
                locations.append(clue.location)

            # locations.append(clue.location)  # for debug


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
