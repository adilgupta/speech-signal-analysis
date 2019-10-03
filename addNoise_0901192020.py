# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:31:26 2017

@author: laya
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  13 14:24:56 2017

@author: ankita
"""
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


########## For expanding noise files ##########
# fileNo = sys.argv[1]
# noise = sys.argv[2]
# startDir = "data/noiseRepo/downSampled/"+noise+"/all/"
# fileName = startDir+noise+fileNo
# [fs,sig]=wav.read(fileName+'.wav')
# N = 6550000
# newNoise(sig,N,fileName,fs)
#################################################

#fileName = sys.argv[1]
num_files_needed = sys.argv[1]

#sys.argv[1]
noise_types = ['babble', 'childrenPlaying', 'rain', 'wind']

clean_files = glob.glob('/media/Sharedata/adil/data/FinalChunks/CSRecordings/GlobalRatingsAudiosNew/*.wav')
clean_files = clean_files[:-50]
num_clean_files = len(clean_files)
clean_files_new = []
for file in clean_files:
    if 'nasik' not in file:
        clean_files_new.append(file)
        #clean_files.remove(file)
clean_files = clean_files_new
num_clean_files = len(clean_files)

for file in clean_files:
    if 'nasik' in file:
        pdb.set_trace()

#noiseNo = sys.argv[4]

#fileName = 'ABX_D_27122016_2_1'#sys.argv[1]
#snr = '10'#sys.argv[2]
#noise = 'rain'#sys.argv[3]
#noiseNo = str(randint(1,12))# sys.argv[4]
#fidcsv=open('info/noisevsAudioMap.csv','a')
#fidcsv.write(fileName+','+noise+','+noiseNo+'\n')
#for fileName in os.listdir('/media/Sharedata/adil/data/FinalChunks/CSRecordings/GlobalRatingsAudiosNew/'):
num_each_type = 10
num_noise_type = 4
SNR_ = 20
#for count in range(int(num_files_needed)):
debug = 0
for c1 in range(int(num_noise_type)):
    for c2 in range(int(num_each_type)):
        snr = SNR_ #np.random.uniform(10, 20)
        temp = np.random.randint(0, 4)
        temp2 = np.random.randint(1, 11)

        if (c1 != 1):
            noiseName = "/media/Sharedata/adil/data/noise/noiseData/" + noise_types[c1] + "/all/" + noise_types[c1] + str(11) + ".wav"
        else:
            noiseName = "/media/Sharedata/adil/data/noise/noiseData/" + noise_types[c1] + "/all/" + "cp" + str(14) + ".wav"
        noise1 = noiseName

        fileName = clean_files[np.random.randint(0, num_clean_files)]
        if('nasik' in fileName):
            #count = count-1
            pdb.set_trace()
            c2 = c2 - 1
            continue
        wavName = fileName #"/media/Sharedata/adil/data/FinalChunks/CSRecordings/GlobalRatingsAudiosNew" + fileName[0:len(fileName)-4] + ".wav"
        # wavName = '../SoundFiles_George/Original/3o33951a.wav'
        labelName = "/media/Sharedata/adil/data/FinalChunks/CSRecordings/GlobalRatingsLabelTracksNew/" + fileName.split('/')[-1][:-4] + ".txt" #fileName[0:len(fileName)-4] + ".txt"
        # labelName = '../SoundFiles_George/1_label_track.txt'
        #noiseName =  noise1 #+ "/" + noise + noiseNo + "_extend.wav ##"/home/swar/swar/shreeharsha/FinalChunks/noise/" +##"
        # noiseName = '../SoundFiles_George/SimRoom3/Noise_SimRoom3_1.wav'
        #saveSigName = "/media/Sharedata/adil/data/FinalChunks/noisy_files/" + fileName.split('/')[-1][:-4] + "_" + noise_types[temp] + "_" + str(snr) + "db.wav"
        saveSigName = "/media/Sharedata/adil/data/test_data/SNR_" + str(snr) + "/" + fileName.split('/')[-1][:-4] + "_" + noise_types[c1] + "_" + str(snr) + "db.wav"
        # saveSigName = '../SoundFiles_George/1_'+snr+'db.wav'

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
        #if(fileName[len(fileName)-5] == '2'): #If the wav file is in 2 parts
        for i in range (0,noLabels):
            if(i==0 and float(lt[0][0])>0): # for the fist unmarked silence
                [sil,noSil,out] = appendSig(0,lt[0][1],sil,noSil,signal,fs,out,0)
            else:  # for the rest of the unmarked silence
                if((float(lt[i][0])-float(lt[i-1][1]))>0):
                    [sil,noSil,out] = appendSig(lt[i-1][1],lt[i][0],sil,noSil,signal,fs,out,0)

        #if(lt[i][2]=='n' or lt[i][2]=='q'): # for the marked silence regions
         #   startSample = np.ceil(float(lt[i][0])*fs)
          #  stopSample = np.floor(float(lt[i][1])*fs)
           # [sil,noSil,out] = appendSig(lt[i][0],lt[i][1],sil,noSil,signal,fs,out,0)
        #else:  # for the speech regions
            [speech,noSpeech,out] = appendSig(lt[i][0],lt[i][1],speech,noSpeech,signal,fs,out,1)

            if(i==noLabels-1 and float(lt[i][1])<totalDuration): # for the last unmarked silence
                [sil,noSil,out] = appendSig(lt[i][1],totalDuration,sil,noSil,signal,fs,out,0)
    #        silDuration = silDuration + (totalDuration - float(lt[i][1]))

        Pspeech = np.sum(np.square(speech)) # power

    # assuming that given signal is clean and the contribution of Psil in Pspeech is negligible
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
        print(debug)
        debug =  debug +1
    # noise = np.int16(noise/np.max(np.abs(noise))*32767)
    # wav.write(saveNoiseName,fs,noise)
