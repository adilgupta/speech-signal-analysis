import numpy as np
from test_audio_091619 import get_output
import glob

test_data_folder = '/media/Sharedata/adil/data/test_data/'
epoch_name = 'generator-100.pkl'
for snr in [10,20]:
    noisy_files = glob.glob(test_data_folder + 'SNR_' + str(snr) + '/*.wav')
    for file in noisy_files:
        get_output(file, epoch_name)
