import numpy as np

from python_speech_features import mfcc


WINLEN = 0.05
WINSTEP = 0.025
SAMPLERATE = 44100

WINLEN_SAMP = round(WINLEN*SAMPLERATE)

def calculate_nfft(samplerate, winlen=0.025):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.

    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.
    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

NFFT = calculate_nfft(SAMPLERATE, winlen=WINLEN)

s = np.zeros(2*WINLEN_SAMP)

feat = mfcc(s, samplerate=SAMPLERATE, winlen=WINLEN, winstep=WINSTEP, nfft=NFFT)

print(np.size(feat))






