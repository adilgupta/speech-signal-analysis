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

snr = 10

noisy_files = glob.glob('./yin_pda-master/nf_folder/SNR_' + str(snr) + '/*.txt')
tmp = []
for file in noisy_files:
    if('kpc' in file):
        tmp.append(file)
noisy_files = tmp
noise_types = ['wind','rain','childrenPlaying','babble']
noisy_f0_errors = np.zeros(4)
enhanced_f0_errors = np.zeros(4)
noisy_counts = np.zeros(4)
enhanced_counts = np.zeros(4)
n_wer, e_wer = [], []
c_c, n_c, e_c = [], [], []

for nf in noisy_files:
    for type in noise_types:
        if type in nf :
            if(type == 'cp'):
                noise_type = 'childrenPlaying'
            else:
                noise_type = type
    cf = './yin_pda-master/cf_folder/' + get_name(nf.split('/')[-1].split('_')[:-2]) + ".txt"#'./Dataset_for_adil/test/' + nf.split('/')[-1].split('_')[:-1] + '.wav'
    ef = './yin_pda-master/ef_folder/SNR_' + str(snr) + '/' + nf.split('/')[-1][:-4] + '_enhanced.txt'
    n_wer_f = './' + str(snr) + 'dbresults/folder/' + nf.split('/')[-1][:12] + noise_type + str(snr) + 'db_' + get_name(nf.split('/')[-1].split('_')[2:6]) + '_csid.csv'
    e_wer_f = './' + str(snr) + 'dbresults/folder/' + nf.split('/')[-1][:12] + noise_type + str(snr) + 'dbenhanced_' + get_name(nf.split('/')[-1].split('_')[2:6]) + '_csid.csv'
############################### remove below
    #cf = './yin_pda-master/cf_folder/test.txt'
############################### remove above
    c = np.loadtxt(cf)
    n = np.loadtxt(nf)
    e = np.loadtxt(ef)



    i = 0
    c_, n_, e_ = [], [], []

    while(i < c.shape[0]):
        num_vals = c[i, 1]
        min_prob = 1;
        if(num_vals == 0):
            c_.append(0)
        else:
            for j in range(int(num_vals)):
                if(min_prob > c[i+j+1,1]):
                    min_prob = c[i+j+1,1]
                    f0 = c[i+j+1,0]
            c_.append(f0)
        i = int(i+1+num_vals)

    i = 0
    while(i < n.shape[0]):
        num_vals = n[i, 1]
        min_prob = 1;
        if(num_vals == 0):
            n_.append(0)
        else:
            for j in range(int(num_vals)):
                if(min_prob > n[i+j+1,1]):
                    min_prob = n[i+j+1,1]
                    f0 = n[i+j+1,0]
            n_.append(f0)
        i = int(i+ num_vals+1)

    i = 0
    while(i < e.shape[0]):
        num_vals = e[i, 1]
        min_prob = 1;
        if(num_vals == 0):
            e_.append(0)
        else:
            for j in range(int(num_vals)):
                if(min_prob > e[i+j+1,1]):
                    min_prob = e[i+j+1,1]
                    f0 = e[i+j+1,0]
            e_.append(f0)
        i = int(i+1+num_vals)



    c_ = np.array(c_)
    e_ = np.array(e_)
    n_ = np.array(n_)
    c_c = np.array(c_c)
    n_c = np.array(n_c)
    e_c = np.array(e_c)

    e_c = np.append(e_c, [np.sqrt(np.mean(np.square(e_c - c_c)))])
    n_c = np.append(n_c, [np.sqrt(np.mean(np.square(n_c - c_c)))])


    noisy_error = np.sqrt(np.mean(np.square(n_[c_!=0] - c_[c_!=0])))
    enhanced_error = np.sqrt(np.mean(np.square(e_[c_!=0] - c_[c_!=0])))
    t = []
    with open(e_wer_f,'r') as f:
        #print(wer_file)
        f2=list(csv.reader(f,delimiter='\n'))
        t=t+f2
    e_wer_tmp = werate(f2)
    t = []
    with open(n_wer_f,'r') as f:
        #print(wer_file)
        f2=list(csv.reader(f,delimiter='\n'))
        t=t+f2
    n_wer_tmp = werate(f2)
    e_wer.append(e_wer_tmp)
    n_wer.append(n_wer_tmp)


    print('done')

    for i in range(4):
        if noise_types[i] in nf:
            noisy_f0_errors[i] += noisy_error
            enhanced_f0_errors[i] += enhanced_error
            noisy_counts[i] += 1
            print(enhanced_f0_errors/noisy_counts)
            print(noisy_error)




    #rmse = np.sqrt(np.sum(np.square(e_[c_ != 0] - c_[c_ != 0])) / len(c_))
    #print(rmse)
    #print(np.sqrt(np.mean(np.square(n_[c_ != 0] - c_[c_ != 0]))))

pdb.set_trace()
