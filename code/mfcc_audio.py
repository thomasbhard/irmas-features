"""Trying to recreate audio from mfcc


"""

import os

import librosa
import soundfile

y, sr = librosa.load(os.path.join(os.path.abspath(__file__), '..', '..', 'audio', 'Velar Prana - Deeper.wav'), offset=30, duration=5)
features = librosa.feature.mfcc(y=y, sr=sr)

audio = librosa.feature.inverse.mfcc_to_audio(features)
soundfile.write(os.path.join(os.path.abspath(__file__), '..', '..', 'audio', 'reconst.wav'), audio, sr)
