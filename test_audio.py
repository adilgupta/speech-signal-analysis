import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from torch.autograd import Variable
from tqdm import tqdm

from data_preprocess import slice_signal, window_size, sample_rate
from model import Generator
from utils import emphasis
import time

#if __name__ == '__main__':
def get_output(FILE_NAME, EPOCH_NAME):
    #parser = argparse.ArgumentParser(description='Test Single Audio Enhancement')
    #parser.add_argument('--file_name', type=str, required=True, help='audio file name')
    #parser.add_argument('--epoch_name', type=str, required=True, help='generator epoch name')

    #opt = parser.parse_args()
    #FILE_NAME = opt.file_name
    #EPOCH_NAME = opt.epoch_name

    generator = nn.DataParallel(Generator())
    generator.load_state_dict(torch.load('/media/Sharedata/adil/epochs_0903191104/' + EPOCH_NAME, map_location='cpu'))
    if torch.cuda.is_available():
        generator.cuda()

    noisy_slices = slice_signal(FILE_NAME, window_size, 1, sample_rate)
    enhanced_speech = []
    for noisy_slice in tqdm(noisy_slices, desc='Generate enhanced audio'):
        z = nn.init.normal(torch.Tensor(1, 1024, 8))
        noisy_slice = torch.from_numpy(emphasis(noisy_slice[np.newaxis, np.newaxis, :])).type(torch.FloatTensor)
        if torch.cuda.is_available():
            noisy_slice, z = noisy_slice.cuda(), z.cuda()
        noisy_slice, z = Variable(noisy_slice), Variable(z)
        t1 = time.time()
        generated_speech = generator(noisy_slice, z).data.cpu().numpy()
        t2 = time.time()
        print('processing_time = ', t2 - t1)
        generated_speech = emphasis(generated_speech, emph_coeff=0.95, pre=False)
        generated_speech = generated_speech.reshape(-1)
        enhanced_speech.append(generated_speech)

    enhanced_speech = np.array(enhanced_speech).reshape(1, -1)
    #file_name = os.path.join(os.path.dirname(FILE_NAME),
    #                         'enhanced_{}.wav'.format(os.path.basename(FILE_NAME).split('.')[0]))
    file_name = '/media/Sharedata/adil/data/test_out/SNR_' + FILE_NAME.split('_')[-1][:-6] + '/' + FILE_NAME.split('/')[-1][:-4] + '_enhanced.wav'
    wavfile.write(file_name, sample_rate, enhanced_speech.T)
