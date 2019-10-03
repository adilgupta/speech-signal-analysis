import numpy
import os
import glob
from scipy.io import wavfile
import pdb

def copy_file(src, dest):
    os.system("sshpass -p 123 scp " + src + " " + dest)
def get_name(a):
    name = ""
    for temp in a:
        name = name + temp + "_"
    name = name[:-1]
    return name

if __name__ == "__main__":
    files = glob.glob("./results092629/test_data_2/SNR_10/*.wav")
    for file in files:
        f_name = ""
        f_name = get_name(file.split('/')[-1].split('_')[:-2]) + ".wav"
        src = "adil@10.107.38.2:/media/Sharedata/adil/data/FinalChunks/CSRecordings/GlobalRatingsAudiosNew/" + f_name
        dest = "./results092629/clean_files_2/SNR_10/" + f_name
        copy_file(src, dest)
