import os
import glob
import random


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


def get_files(path, inst='gel', filt='', ignore=[], rand=False, num=None):
    """Get IRMAS data for specific instrument

    :param path: path to the IRMAS Dataset
    :param inst: specifies the instrument (cel, cla, flu, gac, gel, org, pia, sax, tru, vio, voi)
    :param filt: optinal parameter for filtering additional tags e.g. genre NOTE: returns files that INCLUDE the filt string!
    :param ignore: optional parameter for ignoring certain files, list of string to ignore:
        e.g. ['jaz_blu', 'pop_roc'] for ignoring these genres
    :param rand: optional parameter to shuffle files before limiting
    :param num: optinal parameter for reducing the number of files returned, if None all files are returned

    :return: returns a list of filenames
    """

    path = os.path.join(path, inst, '*.wav')


    files = glob.glob(path) # get all files for specified instrument
    files_filtered = [f for f in files if filt in f] # collect all files that contain filt

    files_ignore = [f for f in files_filtered if not any(tag in f for tag in ignore)]

    if rand:
        random.shuffle(files_ignore)

    num_files = len(files_ignore)

    # optional reduction if num is specified
    if num is not None and num < num_files:
        files_ignore = files_ignore[:num]
        num_files = num
    
    print(str(num_files) + ' Files from folder ' + inst + ' with filter: ' + filt)

    return files_ignore

def get_label(filename):
    """Get label from full filename

    :param filename: filename with path
    
    :return: returns label e.g. 'pia' or 'gel'
    """
    basename = os.path.basename(filename)

    label = basename.split('[', 1)[1]
    label = label.split(']',1)[0]

    return label


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()



if __name__ == "__main__":
    IRMAS_PATH = 'C:\\Users\\thoma\\Documents\\_STUDIUM\\5. Semester\\Fachvertiefung Software\\IRMAS-TrainingData'

    filenames = get_files(IRMAS_PATH, inst='cel', rand=True, num=5)
    
    print([os.path.basename(f) for f in filenames])
    print(len(filenames))

    