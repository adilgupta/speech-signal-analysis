import os
import numpy as np
from numpy import *
import sys
import csv
import scipy.io.wavfile as wav
import scipy.signal as sigpy
from random import randint
import glob
import pdb

def appendSig(startTime,stopTime,sig,noSig,signal,fs,out,value):

    startSample = int(np.ceil(float(startTime)*fs))
    stopSample = int(np.floor(float(stopTime)*fs))
    sig = np.append(sig,signal[startSample:stopSample])
    noSig = noSig + (stopSample-startSample)
    out[startSample:stopSample] = value
    return sig,noSig,out

def findInterval(noise,lenSig,lenNoise):
    sig = []
    if(lenSig<=lenNoise):
        start = randint(0,lenNoise-lenSig)
        sig = noise[start:start+lenSig]
    else:
        length=0
        # while((length+lenNoise)<lenSig):
        while(1):
            start = randint(0,lenNoise)
#            sig = np.append(sig,noise[start:lenNoise])
#            length = length + (lenNoise-start)
            end = randint(start,lenNoise)
            sig = np.append(sig,noise[start:end])
            length = length + (end-start)
            extra = lenSig - length
            if(extra<0):
                break
        start = randint(0, len(sig) - lenSig) #randint(0,lenNoise-extra)
        sig = sig[start:start + lenSig]
    return sig

def newNoise(sig,N,fileName,fs):  # for extending noise files, so that len(noise)>len(sig) as far as possible
# though this code also handles the case where len(noise)<len(signal)

    addN = N/len(sig) + 1
    newSig=[]
    for j in range (0,addN):
        newSig=np.append(newSig,sig)
    newSig=np.int16(newSig/np.max(np.abs(newSig))*32767)
    wav.write(fileName+'_extend.wav',fs,newSig)
    return newSig


#sys.argv[1]
noise_types = ['babble', 'childrenPlaying', 'rain', 'wind']

clean_files = glob.glob('/media/Sharedata/adil/atmpt_1008190009/Dataset_for_adil/train/*.wav')
clean_files = clean_files
num_clean_files = len(clean_files)

for noise_tp in noise_types:
    i = 0
    for cf in clean_files:
        snr = 10 + 5*(i%20) / 10
        i+=1
        temp2 = np.random.randint(1, 11)
        if (noise_tp != 'childrenPlaying'):
            noiseName = "/media/Sharedata/adil/data/noise/noiseData/" + noise_tp + "/all/" + noise_types[temp] + str(temp2) + ".wav"
        else:
            noiseName = "/media/Sharedata/adil/data/noise/noiseData/" + noise_tp + "/all/" + "cp" + str(temp2 + 1) + ".wav"

        wavName = cf

        labelName = "/media/Sharedata/adil/data/FinalChunks/CSRecordings/GlobalRatingsLabelTracksNew/" + wavName.split('/')[-1][:-4] + ".txt"
        saveSigName = "/media/Sharedata/adil/atmpt_1008190009/train_dataset/" + wavName.split('/')[-1][:-4] + "_" + noise_tp + "_" + str(snr) + "db.wav"

        [fs, signal] = wav.read(wavName)
        [fsn, noise] = wav.read(noiseName)
        if (noise.shape[-1] == 2):
            noise = noise[:,0]
        if(fs!=fsn):
            print ("invalid sampling rate")
        lenNoise = len(noise)
        lenSig = len(signal)
        if(lenSig>lenNoise):
            print ("noise file length not sufficient")
        signal = signal/float(np.max(signal))
        noise = noise/float(np.max(noise))
        #signal = signal/30000.0
        #noise = noise/30000.0
        totalDuration = len(signal)/float(fs)

        fid = open(labelName,"r")
        print(labelName)
        lt = list(csv.reader(fid,delimiter="\t"))
        noLabels = len(lt)
        sil = []
        speech = []
        noSil = 0
        noSpeech = 0
        snr = float(snr)
        out = np.zeros(lenSig) # labels
        for i in range (0,len(lt)):
            lt[i][0] = str(float(lt[i][0]) - float(lt[0][0]))
            lt[i][1] = str(float(lt[i][1]) - float(lt[0][0]))

        for i in range (0,noLabels):
            if(i==0 and float(lt[0][0])>0): # for the fist unmarked silence
                [sil,noSil,out] = appendSig(0,lt[0][1],sil,noSil,signal,fs,out,0)
            else:  # for the rest of the unmarked silence
                if((float(lt[i][0])-float(lt[i-1][1]))>0):
                    [sil,noSil,out] = appendSig(lt[i-1][1],lt[i][0],sil,noSil,signal,fs,out,0)
            [speech,noSpeech,out] = appendSig(lt[i][0],lt[i][1],speech,noSpeech,signal,fs,out,1)

            if(i==noLabels-1 and float(lt[i][1])<totalDuration): # for the last unmarked silence
                [sil,noSil,out] = appendSig(lt[i][1],totalDuration,sil,noSil,signal,fs,out,0)

        Pspeech = np.sum(np.square(speech))

        P_addedNoise = Pspeech*pow(10,(-1*snr/10)) # noise power for the required SNR
        # start = randint(0,lenNoise-lenSig)
        # noise = noise[start:start+lenSig]
        noise = findInterval(noise,lenSig,lenNoise)
        if(np.shape(noise) != np.shape(out)):
            pdb.set_trace()
        scalingFactor = np.sqrt(P_addedNoise)/np.sqrt(np.sum(np.square(noise*out)))
        noise = scalingFactor * noise

        noisySig = noise + signal

        noisySig = np.int16(noisySig/np.max(np.abs(noisySig))*32767)
        wav.write(saveSigName,fs,noisySig)
