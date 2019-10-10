import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pdb
import crepe
import os, csv

noisy_files = glob.glob('./../Dataset_for_adil/noisy_files_test/SNR_10/*.wav')
for nf in noisy_files:
    cf = './../Dataset_for_adil/test/' + get_name(nf.split('/')[-1].split('_')[:-2]) + ".wav"#'./Dataset_for_adil/test/' + nf.split('/')[-1].split('_')[:-1] + '.wav'
    ef = './../Dataset_for_adil/test_out/SNR_' + str(snr) + '/' + nf.split('/')[-1][:-4] + '_enhanced.wav'
