import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pdb
import crepe
import os

def get_name(a):
    name = ""
    for temp in a:
        name = name + temp + "_"
    name = name[:-1]
    return name

if __name__ == '__main__':
    snr = 10
    for n_type in ['rain', 'babble', 'childrenPlaying', 'wind']:
        nf = './Dataset_for_adil/noisy_files_test/SNR_10/kpc_27122016_1_cs_d1_1_babble_10db.wav'
        cf = './Dataset_for_adil/test/' + get_name(nf.split('/')[-1].split('_')[:-2]) + ".wav"#'./Dataset_for_adil/test/' + nf.split('/')[-1].split('_')[:-1] + '.wav'
        ef = './Dataset_for_adil/test_out/SNR_' + str(snr) + '/' + nf.split('/')[-1][:-4] + '_enhanced.wav'

        fs, noisy = wavfile.read(nf)
        fs, clean = wavfile.read(cf)
        fs, enhanced = wavfile.read(ef)

        print(clean.dtype)
        print(noisy.dtype)

        time_clean, frequency_clean, confidence_clean, activation_clean = crepe.predict(clean, fs, viterbi=True)
        time_noisy, frequency_noisy, confidence_noisy, activation_noisy = crepe.predict(noisy, fs, viterbi=True)
        time_enhanced, frequency_enhanced, confidence_enhanced, activation_enhanced = crepe.predict(enhanced, fs, viterbi=True)

        print(np.sqrt(np.mean(np.square(frequency_enhanced[confidence_clean>0.6] - frequency_clean[confidence_clean>0.6]))))
        print(np.sqrt(np.mean(np.square(frequency_noisy[confidence_clean>0.6] - frequency_clean[confidence_clean>0.6]))))
        pdb.set_trace()
