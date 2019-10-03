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

if __name__ == "__main__":
    snr_ = [10, 20]
    i = 0
    for snr in snr_:
        noisy_files = glob.glob('./Dataset_for_adil/noisy_files_test/SNR_'+ str(snr) +'/*.wav')
        for nf in noisy_files:
            print(i)
            i+=1
            cf = './Dataset_for_adil/test/' + get_name(nf.split('/')[-1].split('_')[:-2]) + ".wav"#'./Dataset_for_adil/test/' + nf.split('/')[-1].split('_')[:-1] + '.wav'
            ef = './Dataset_for_adil/test_out/SNR_' + str(snr) + '/' + nf.split('/')[-1][:-4] + '_enhanced.wav'
            fs, noisy = wavfile.read(nf)
            fs, clean = wavfile.read(cf)
            fs, enhanced = wavfile.read(ef)

            if(len(clean) > len(enhanced) or len(noisy) > len(enhanced)):
                clean = clean[:len(enhanced)]
                noisy = noisy[:len(enhanced)]
                wavfile.write(cf, data = clean, rate = fs)
                wavfile.write(nf, data = noisy, rate = fs)
