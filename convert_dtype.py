from scipy.io import wavfile
import glob
import numpy as np



if __name__ == '__main__':
    fls = glob.glob('/media/Sharedata/adil/atmpt_1008190009/Dataset_for_adil/test/*.wav')
    for file in fls:
        fs, data = wavfile.read(fls)
        if(data.dtype == 'int16'):
            data = data.astype(np.float64) / np.iinfo(data.dtype).max
            wavfile.write(file, data, fs)
            print('converted')

            
