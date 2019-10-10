import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pdb
import crepe
import os, csv
from pesq_analysis import _sqrt, melcd

snr = 10
_logdb_const = 10.0 / np.log(10.0) * np.sqrt(2.0)

def get_name(a):
    name = ""
    for temp in a:
        name = name + temp + "_"
    name = name[:-1]
    return name

def werate(word):
	tot_count = 0
	l=0
	for j in range(len(word)):

		if(word[j][0] == 'c'):
			tot_count+=1
		elif(word[j][0] == 's'):
			tot_count+=1
			l+=1
		elif(word[j][0] == 'd'):
			tot_count+=1
			l+=1
		elif(word[j][0]=='i'):
			l+=1

	return l/(0.0+tot_count)

if __name__ == '__main__':
    noisy_files = glob.glob('./Dataset_for_adil/noisy_files_test/SNR_10/*.wav')
    df = pd.DataFrame(columns = ('file_name', 'f0_error_n', 'f0_error_e', 'average_con_n', 'average_con_c', 'average_con_e', 'con_error_n', 'con_error_e', 'wer_e', 'wer_n'))
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

        time_clean, frequency_clean, confidence_clean, activation_clean = crepe.predict(clean, fs, viterbi=True)
        time_noisy, frequency_noisy, confidence_noisy, activation_noisy = crepe.predict(noisy, fs, viterbi=True)
        time_enhanced, frequency_enhanced, confidence_enhanced, activation_enhanced = crepe.predict(enhanced, fs, viterbi=True)

        f0_error_n = np.sqrt(np.mean(np.square(frequency_noisy[confidence_clean > 0.5] - frequency_clean[confidence_clean>0.5])))
        f0_error_e = np.sqrt(np.mean(np.square(frequency_enhanced[confidence_clean > 0.5] - frequency_clean[confidence_clean>0.5])))
        avg_con_n = np.mean(confidence_noisy)
        avg_con_c = np.mean(confidence_clean)
        avg_con_e = np.mean(confidence_enhanced)

        con_error_n = np.sqrt(np.mean(np.square(confidence_noisy - confidence_clean)))
        con_error_e = np.sqrt(np.mean(np.square(confidence_enhanced - confidence_clean)))

        t = []
        with open(enhanced_wer_file,'r') as f:
            print(enhanced_wer_file)
            f2=list(csv.reader(f,delimiter='\n'))
            t=t+f2
        wer_e = werate(f2)

        t = []
        with open(noisy_wer_file,'r') as f:
            print(noisy_wer_file)
            f2=list(csv.reader(f,delimiter='\n'))
            t=t+f2
        wer_n = werate(f2)

        df = df.append({'file_name':nf, 'f0_error_n':f0_error_n, 'f0_error_e':f0_error_e, 'average_con_n':avg_con_n, 'average_con_c':avg_con_c, 'average_con_e':avg_con_e, 'con_error_n':con_error_n, 'con_error_e':con_error_e, 'wer_e':wer_e, 'wer_n':wer_n}, ignore_index = True)
        df.to_csv('df_1006101930.csv')
    pdb.set_trace()
