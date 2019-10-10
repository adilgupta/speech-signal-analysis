import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pdb
import crepe
import os, csv


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

if __name__ == "__main__":
    snr = 20
    noisy_wers = {'babble':0, 'childrenPlaying':0, 'rain':0, 'wind':0}
    noisy_counts = {'babble':0, 'childrenPlaying':0, 'rain':0, 'wind':0}
    noisy_files = glob.glob('./Dataset_for_adil/noisy_files_test/SNR_20/*.wav')
    for nf in noisy_files:
        noise_types = ['babble', 'childrenPlaying', 'rain', 'wind']
        for type in noise_types:
            if type in nf :
                if(type == 'childrenPlaying'):
                    noise_type = 'childrenPlaying'
                else:
                    noise_type = type
        cf = './Dataset_for_adil/test/' + get_name(nf.split('/')[-1].split('_')[:-2]) + ".wav"#'./Dataset_for_adil/test/' + nf.split('/')[-1].split('_')[:-1] + '.wav'
        ef = './Dataset_for_adil/test_out/SNR_' + str(snr) + '/' + nf.split('/')[-1][:-4] + '_enhanced.wav'
        wer_file =  './20dbresults/'+ nf.split('/')[-1][:12] + noise_type + str(snr) + 'db_' + get_name(nf.split('/')[-1].split('_')[2:6]) + '_csid.csv'
        t = []

        with open(wer_file,'r') as f:
            print(wer_file)
            f2=list(csv.reader(f,delimiter='\n'))
            t=t+f2
        wer_temp = werate(f2)
        noisy_wers[noise_type] = (noisy_counts[noise_type]*noisy_wers[noise_type] + wer_temp)/(noisy_counts[noise_type]+1)
        noisy_counts[noise_type]+=1
    pdb.set_trace()
