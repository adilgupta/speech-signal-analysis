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
    snr = 10
    noisy_files = glob.glob('./Dataset_for_adil/noisy_files_test/SNR_'+ str(snr) +'/*.wav')
    if(not os.path.exists('./crepe_pitch_analysis/test')):
        os.mkdir('./crepe_pitch_analysis/test')
    if(not os.path.exists('./crepe_pitch_analysis/test/time')):
        os.mkdir('./crepe_pitch_analysis/test/time')
        os.mkdir('./crepe_pitch_analysis/test/frequency')
        os.mkdir('./crepe_pitch_analysis/test/confidence')
        os.mkdir('./crepe_pitch_analysis/test/activation')
    if(not os.path.exists('./crepe_pitch_analysis/test_out')):
        os.mkdir('./crepe_pitch_analysis/test_out')
        os.mkdir('./crepe_pitch_analysis/test_out/SNR_10')
        os.mkdir('./crepe_pitch_analysis/test_out/SNR_20')
    if(not os.path.exists('./crepe_pitch_analysis/test_out/time')):
        os.mkdir('./crepe_pitch_analysis/test_out/time')
        os.mkdir('./crepe_pitch_analysis/test_out/frequency')
        os.mkdir('./crepe_pitch_analysis/test_out/confidence')
        os.mkdir('./crepe_pitch_analysis/test_out/activation')
    if(not os.path.exists('./crepe_pitch_analysis/noisy_files_test')):
        os.mkdir('./crepe_pitch_analysis/noisy_files_test')
        os.mkdir('./crepe_pitch_analysis/noisy_files_test/SNR_20')
        os.mkdir('./crepe_pitch_analysis/noisy_files_test/SNR_10')
    if(not os.path.exists('./crepe_pitch_analysis/noisy_files_test/time')):
        os.mkdir('./crepe_pitch_analysis/noisy_files_test/time')
        os.mkdir('./crepe_pitch_analysis/noisy_files_test/frequency')
        os.mkdir('./crepe_pitch_analysis/noisy_files_test/confidence')
        os.mkdir('./crepe_pitch_analysis/noisy_files_test/activation')

    for nf in noisy_files:
        cf = './Dataset_for_adil/test/' + get_name(nf.split('/')[-1].split('_')[:-2]) + ".wav"#'./Dataset_for_adil/test/' + nf.split('/')[-1].split('_')[:-1] + '.wav'
        ef = './Dataset_for_adil/test_out/SNR_' + str(snr) + '/' + nf.split('/')[-1][:-4] + '_enhanced.wav'
        fs, noisy = wavfile.read(nf)
        fs, clean = wavfile.read(cf)
        fs, enhanced = wavfile.read(ef)
        print(nf)
        if(len(clean) > len(enhanced) or len(noisy) > len(enhanced)):
            clean = clean[:len(enhanced)]
            noisy = noisy[:len(enhanced)]
            wavfile.write(cf, data = clean, rate = fs)
            wavfile.write(nf, data = noisy, rate = fs)

        time_clean, frequency_clean, confidence_clean, activation_clean = crepe.predict(clean, fs, viterbi=True)
        time_noisy, frequency_noisy, confidence_noisy, activation_noisy = crepe.predict(noisy, fs, viterbi=True)
        time_enhanced, frequency_enhanced, confidence_enhanced, activation_enhanced = crepe.predict(enhanced, fs, viterbi=True)
        #clean_pitch = crepe.predict(clean, fs, viterbi=True)
        np.save('./crepe_pitch_analysis/test/time/' + cf.split('/')[-1][:-4] + '.npy', time_clean)
        np.save('./crepe_pitch_analysis/test/frequency/' + cf.split('/')[-1][:-4] + '.npy', frequency_clean)
        np.save('./crepe_pitch_analysis/test/confidence/' + cf.split('/')[-1][:-4] + '.npy', confidence_clean)
        np.save('./crepe_pitch_analysis/test/activation/' + cf.split('/')[-1][:-4] + '.npy', activation_clean)

        np.save('./crepe_pitch_analysis/test_out/time/' + cf.split('/')[-1][:-4] + '.npy', time_enhanced)
        np.save('./crepe_pitch_analysis/test_out/frequency/' + cf.split('/')[-1][:-4] + '.npy', frequency_enhanced)
        np.save('./crepe_pitch_analysis/test_out/confidence/' + cf.split('/')[-1][:-4] + '.npy', confidence_enhanced)
        np.save('./crepe_pitch_analysis/test_out/activation/' + cf.split('/')[-1][:-4] + '.npy', activation_enhanced)

        np.save('./crepe_pitch_analysis/noisy_files_test/time/' + cf.split('/')[-1][:-4] + '.npy', time_noisy)
        np.save('./crepe_pitch_analysis/noisy_files_test/frequency/' + cf.split('/')[-1][:-4] + '.npy', frequency_noisy)
        np.save('./crepe_pitch_analysis/noisy_files_test/confidence/' + cf.split('/')[-1][:-4] + '.npy', confidence_noisy)
        np.save('./crepe_pitch_analysis/noisy_files_test/activation/' + cf.split('/')[-1][:-4] + '.npy', activation_noisy)


        pdb.set_trace()
