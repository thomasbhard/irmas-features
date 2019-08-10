import os
import glob


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


def get_files(path, inst='gel', filt='', num=None):
    """Get IRMAS data for specific instrument

    :param path: path to the IRMAS Dataset
    :param inst: specifies the instrument (cel, cla, flu, gac, gel, org, pia, sax, tru, vio, voi)
    :param filt: optinal parameter for filtering additional tags e.g. genre
    :param num: optinal parameter for reducing the number of files returned, if None all files are returned

    :return: returns a list of filenames
    """

    path = os.path.join(path, inst, '*.wav')

    files = glob.glob(path)
    files_filtered = [file for file in files if filt in file]

    num_files = len(files_filtered)

    # optional reduction if num is specified
    if num is not None and num < num_files:
        files_filtered = files_filtered[:num]
        num_files = num
    
    print(str(num_files) + ' Files from folder ' + inst + ' with filter: ' + filt)

    return files_filtered



if __name__ == "__main__":
    pass

    