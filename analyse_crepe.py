import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pdb
import crepe
import os, csv
from pesq_analysis import _sqrt, melcd

def get_name(a):
    name = ""
    for temp in a:
        name = name + temp + "_"
    name = name[:-1]
    return name

if __name__ == '__main__':
    snr = 10
    noisy_files = glob.glob('./Dataset_for_adil/noisy_files_test/SNR_10/*.wav')
    for nf in noisy_files:

        noise_types = ['babble', 'cp', 'rain', 'wind']
        for type in noise_types:
            if type in nf :
                if(type == 'cp'):
                    noise_type = 'childrenPlaying'
                else:
                    noise_type = type

        cf = './Dataset_for_adil/test/' + get_name(nf.split('/')[-1].split('_')[:-2]) + ".wav"#'./Dataset_for_adil/test/' + nf.split('/')[-1].split('_')[:-1] + '.wav'
        ef = './Dataset_for_adil/test_out/SNR_' + str(snr) + '/' + nf.split('/')[-1][:-4] + '_enhanced.wav'
        enhanced_wer_file =  './10dbresults/folder/'+ nf.split('/')[-1][:12] + noise_type + str(snr) + 'dbenhanced_' + get_name(nf.split('/')[-1].split('_')[2:6]) + '_csid.csv'
        noisy_wer_file = './10dbresults/folder/'+ nf.split('/')[-1][:12] + noise_type + str(snr) + 'db_' + get_name(nf.split('/')[-1].split('_')[2:6]) + '_csid.csv'

        fs, clean = wavfile.read(cf)
        fs, noisy = wavfile.read(nf)
        fs, enhanced = wavfile.read(ef)

        if(clean.dtype == 'int16'):
            clean = clean.astype(np.float64) / np.iinfo(clean.dtype).max
            wavfile.write(cf, data = clean, rate = fs)
            print('there it is')
        if(noisy.dtype == 'int16'):
            noisy = noisy.astype(np.float64) / np.iinfo(noisy.dtype).max
            wavfile.write(nf, data = noisy, rate = fs)

        #time_clean, frequency_clean, confidence_clean, activation_clean = crepe.predict(clean, fs, viterbi=True)
        #time_noisy, frequency_noisy, confidence_noisy, activation_noisy = crepe.predict(noisy, fs, viterbi=True)
        #time_enhanced, frequency_enhanced, confidence_enhanced, activation_enhanced = crepe.predict(enhanced, fs, viterbi=True)

        pdb.set_trace()

        salience = np.flip(activation, axis=1)

        inferno = matplotlib.cm.get_cmap('inferno')
        image = inferno(salience.transpose())
