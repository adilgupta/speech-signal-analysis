import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

df = pd.read_csv('df_1006101930.csv')


con_err_e, con_err_n = [], []
con_avg_e, con_avg_n, con_avg_c = [],[],[]
wer_e, wer_n = [],[]

num_fls = np.zeros(4)
noise_types = ['babble', 'childrenPlaying', 'rain', 'wind']


for i in range(len(df)):
    if('kpc' in df['file_name'][i]):
        con_err_e.append(df['con_error_e'][i])
        con_err_n.append(df['con_error_n'][i])
        wer_e.append(df['wer_e'][i])
        wer_n.append(df['wer_n'][i])
        for j in range(4):
            if noise_types[j] in df['file_name'][i]:
                num_fls[j]+=1
con_err_e = np.array(con_err_e)
con_err_n = np.array(con_err_n)
wer_e = np.array(wer_e)
wer_n = np.array(wer_n)

print(np.corrcoef((con_err_e - con_err_n), (wer_e - wer_n)))
print(num_fls)
print(len(df))

num_each_noise = np.zeros(4)
f0_err = np.zeros(4)
print(df.columns)

for i in range(len(df)):
    for j in range(4):
        if noise_types[j] in df['file_name'][i]:
            num_each_noise[j] += 1
            f0_err[j] += df['f0_error_e'][i]
print(f0_err / num_each_noise)
print(noise_types)
