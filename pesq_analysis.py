import numpy
import os
import glob
from scipy.io import wavfile
from pypesq import pypesq
from python_speech_features import mfcc
import pandas as pd
import numpy as np
import math
import pdb
import matlab.engine
eng = matlab.engine.start_matlab()

snr = 10
_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)

def get_name(a):
    name = ""
    for temp in a:
        name = name + temp + "_"
    name = name[:-1]
    return name

def _sqrt(x):
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return np.sqrt(x) if isnumpy else math.sqrt(x) if isscalar else x.sqrt()

def melcd(X, Y, lengths=None):
    """Mel-cepstrum distortion (MCD).

    The function computes MCD for time-aligned mel-cepstrum sequences.

    Args:
        X (ndarray): Input mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        Y (ndarray): Target mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.

    Returns:
        float: Mean mel-cepstrum distortion in dB.

    .. note::

        The function doesn't check if inputs are actually mel-cepstrum.
    """
    # summing against feature axis, and then take mean against time axis
    # Eq. (1a)
    # https://www.cs.cmu.edu/~awb/papers/sltu2008/kominek_black.sltu_2008.pdf
    if lengths is None:
        z = X - Y
        r = _sqrt((z * z).sum(-1))
        if not np.isscalar(r):
            r = r.mean()
        return _logdb_const * float(r)

    # Case for 1-dim features.
    if len(X.shape) == 2:
        # Add feature axis
        X, Y = X[:, :, None], Y[:, :, None]

    s = 0.0
    T = _sum(lengths)
    for x, y, length in zip(X, Y, lengths):
        x, y = x[:length], y[:length]
        z = x - y
        s += _sqrt((z * z).sum(-1)).sum()

    return _logdb_const * float(s) / float(T)


if __name__ == "__main__":
    noisy_files = glob.glob('./Dataset_for_adil/noisy_files_test/SNR_' + str(snr)+ '/*.wav')
    #pdb.set_trace()
    noise_types = ['rain', 'babble', 'childrenPlaying', 'wind']
    inp_pesq = [0,0,0,0]
    out_pesq = [0,0,0,0]
    num_files = [0,0,0,0]
    #pesq_results = pd.DataFrame(columns = ['noise_type', 'inp_pesq', 'out_pesq'])
    #pdb.set_trace()
    for file in noisy_files:
        noisy_file = file
        clean_file = './Dataset_for_adil/test' + '/' + get_name(file.split('/')[-1].split('_')[:-2]) + '.wav'
        enhanced_file = './Dataset_for_adil/test_out/SNR_' + str(snr) + '/' + file.split('/')[-1][:-4] + '_enhanced.wav'

        fs, noisy = wavfile.read(noisy_file)
        fs, clean = wavfile.read(clean_file)
        fs, enhanced = wavfile.read(enhanced_file)

        if (len(clean) > len(enhanced) or len(noisy) > len(enhanced)):
            clean = clean[:len(enhanced)]
            noisy = noisy[:len(enhanced)]
            wavfile.write(clean_file, data = clean, rate = fs)
            wavfile.write(noisy_file, data = noisy, rate = fs)

        if(len(clean) != len(noisy)):
            print("what the hell")
            pdb.set_trace()

        clean = clean.astype(np.float) / np.iinfo(clean.dtype).max
        noisy = noisy.astype(np.float) / np.iinfo(noisy.dtype).max

        eng.workspace['clean'] = eng.audioread(clean_file)
        eng.workspace['noisy'] = eng.audioread(noisy_file)
        eng.workspace['enhanced'] = eng.audioread(enhanced_file)
        eng.workspace['fs'] = 16000.0

        #pdb.set_trace()
        mfcc_clean = eng.mfcc(eng.workspace['clean'], eng.workspace['fs'], 'Numcoeffs', 10.)#eng.mfcc(matlab.double(clean.tolist()), matlab.double(16000))#eng.mfcc(clean.tolist, 16000)#, numcep = 14)
        mfcc_noisy = eng.mfcc(eng.workspace['noisy'], eng.workspace['fs'], 'Numcoeffs', 10.)   # mfcc(noisy, 16000, numcep = 14)
        mfcc_enhanced = eng.mfcc(eng.workspace['enhanced'], eng.workspace['fs'], 'Numcoeffs', 10.)#mfcc(enhanced, 16000, numcep = 14)
        #pdb.set_trace()
        inp_pesq_ = melcd(np.array(mfcc_clean)[:,1:], np.array(mfcc_noisy)[:,1:], lengths=None) # pypesq(fs, clean, noisy, 'nb')
        out_pesq_ = melcd(np.array(mfcc_clean)[:,1:], np.array(mfcc_enhanced)[:,1:], lengths=None) # pypesq(fs, clean, enhanced, 'nb')
        #pdb.set_trace()
        for i in range(4):
            if(noise_types[i] in noisy_file):
                inp_pesq[i] = (num_files[i] * inp_pesq[i] + inp_pesq_) / (num_files[i] + 1)
                out_pesq[i] = (num_files[i] * out_pesq[i] + out_pesq_) / (num_files[i] + 1)
                num_files[i] += 1

        #print(inp_pesq, out_pesq)
        print(inp_pesq)
        print(out_pesq)
    pdb.set_trace()
