from yin import main
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pdb
import os

def get_name(a):
    name = ""
    for temp in a:
        name = name + temp + "_"
    name = name[:-1]
    return name

if __name__ == "__main__":
    snr = 10
    noisy_files = glob.glob('./Dataset_for_adil/noisy_files_test/SNR_10/kpc_27122016_1_cs_f1_1_*.wav')

    for nf in noisy_files:
        cf = './Dataset_for_adil/test/' + get_name(nf.split('/')[-1].split('_')[:-2]) + ".wav"#'./Dataset_for_adil/test/' + nf.split('/')[-1].split('_')[:-1] + '.wav'
        ef = './Dataset_for_adil/test_out/SNR_' + str(snr) + '/' + nf.split('/')[-1][:-4] + '_enhanced.wav'

        fs, clean = wavfile.read(cf)
        fs, noisy = wavfile.read(nf)
        fs, enhanced = wavfile.read(ef)

        clean = clean.astype(np.float64) / np.iinfo(clean.dtype).max
        noisy = noisy.astype(np.float64) / np.iinfo(noisy.dtype).max

        pitches_c, harmonic_rates_c, argmins_c, times_c = main(audioFileName = cf, audioDir = None)
        pitches_n, harmonic_rates_n, argmins_n, times_n = main(audioFileName = nf, audioDir = None)
        pitches_e, harmonic_rates_e, argmins_e, times_e = main(audioFileName = ef, audioDir = None)

        pitches_c = np.array(pitches_c)
        pitches_n = np.array(pitches_n)
        pitches_e = np.array(pitches_e)

        np.sqrt(np.mean(np.square(pitches_n[np.logical_and(pitches_c!=0, pitches_n!=0)] - pitches_c[np.logical_and(pitches_c!=0, pitches_n!=0)])))

        #np.sqrt(np.mean(pitches_n[pitches_n!=0 and pitches_c!=0] - pitches_c[pitches_n!=0 and pitches_c!=0]))


        pdb.set_trace()
