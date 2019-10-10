import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pdb
import crepe
import os, csv
import matlab.engine
from pesq_analysis import _sqrt, melcd
eng = matlab.engine.start_matlab()

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
    df = pd.DataFrame(columns = ('noisy_file','noisy_f0_error','enhanced_f0_error','noisy_cep_dist','enhanced_cep_dist','voicing_clean','voicing_noisy','voicing_enhanced'))
    df_2 = pd.DataFrame(columns = ('file', 'noisy_f0_error', 'enhanced_f0_error', 'noisy_cep_dist', 'enhanced_cep_dist', 'wer_e', 'wer_n'))
    tmp = []
    for file in noisy_files:
        if 'babble' in file:
            tmp.append(file)
    noisy_files = tmp

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

        #clean = clean.astype(np.float64) / np.iinfo(clean.dtype).max
        #noisy = noisy.astype(np.float64) / np.iinfo(noisy.dtype).max

        time_clean, frequency_clean, confidence_clean, activation_clean = crepe.predict(clean, fs, viterbi=True)
        time_noisy, frequency_noisy, confidence_noisy, activation_noisy = crepe.predict(noisy, fs, viterbi=True)
        time_enhanced, frequency_enhanced, confidence_enhanced, activation_enhanced = crepe.predict(enhanced, fs, viterbi=True)


        enhanced_f0_error = np.sqrt(np.mean(np.square(frequency_enhanced[confidence_clean>0.5] - frequency_clean[confidence_clean>0.5])))
        noisy_f0_error = np.sqrt(np.mean(np.square(frequency_noisy[confidence_clean>0.5] - frequency_clean[confidence_clean>0.5])))

        total_len = len(time_enhanced)


        eng.workspace['clean'] = eng.audioread(cf)
        eng.workspace['noisy'] = eng.audioread(nf)
        eng.workspace['enhanced'] = eng.audioread(ef)
        eng.workspace['fs'] = 16000.0

        mfcc_clean = eng.mfcc(eng.workspace['clean'], eng.workspace['fs'], 'Numcoeffs', 10.)#eng.mfcc(matlab.double(clean.tolist()), matlab.double(16000))#eng.mfcc(clean.tolist, 16000)#, numcep = 14)
        mfcc_noisy = eng.mfcc(eng.workspace['noisy'], eng.workspace['fs'], 'Numcoeffs', 10.)   # mfcc(noisy, 16000, numcep = 14)
        mfcc_enhanced = eng.mfcc(eng.workspace['enhanced'], eng.workspace['fs'], 'Numcoeffs', 10.)

        inp_cep_ = melcd(np.array(mfcc_clean)[:,1:], np.array(mfcc_noisy)[:,1:], lengths=None) # pypesq(fs, clean, noisy, 'nb')
        out_cep_ = melcd(np.array(mfcc_clean)[:,1:], np.array(mfcc_enhanced)[:,1:], lengths=None)

        voicing_clean = np.sum(confidence_clean[confidence_clean>0.5])/np.sum(confidence_clean[confidence_clean<0.5])
        voicing_noisy = np.sum(confidence_noisy[confidence_noisy>0.5])/np.sum(confidence_noisy[confidence_noisy<0.5])
        voicing_enhanced = np.sum(confidence_enhanced[confidence_enhanced>0.5])/np.sum(confidence_enhanced[confidence_enhanced<0.5])

        df = df.append({'noisy_file':nf, 'noisy_f0_error':noisy_f0_error, 'enhanced_f0_error':enhanced_f0_error, 'noisy_cep_dist':inp_cep_, 'enhanced_cep_dist':out_cep_, 'voicing_clean':voicing_clean, 'voicing_noisy':voicing_noisy, 'voicing_enhanced':voicing_enhanced}, ignore_index = True)
        t = []

        with open(enhanced_wer_file,'r') as f:
            #print(wer_file)
            f2=list(csv.reader(f,delimiter='\n'))
            t=t+f2
        en_wer = werate(f2)

        t = []
        with open(noisy_wer_file,'r') as f:
            #print(wer_file)
            f2=list(csv.reader(f,delimiter='\n'))
            t=t+f2
        n_wer = werate(f2)

        df_2 = df_2.append({'noisy_file':nf, 'noisy_f0_error':noisy_f0_error, 'enhanced_f0_error':enhanced_f0_error, 'noisy_cep_dist':inp_cep_, 'enhanced_cep_dist':out_cep_, 'wer_e':en_wer, 'wer_n':n_wer}, ignore_index = True)

        #print(np.corrcoef(np.array(df_2['enhanced_f0_error']), np.array(df_2['wer'])), np.corrcoef(np.array(df_2['enhanced_cep_dist']), np.array(df_2['wer'])))

        pdb.set_trace()
