# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:55:22 2020

@author: Shuning Liu
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import sys
import array
import math
import os
from matplotlib.colors import LogNorm
import argparse


parser = argparse.ArgumentParser(description='A rolling spectrogram')
parser.add_argument("--file",default='eeg.bin',help="--file='path_to_file'")
parser.add_argument("--psdMin",default=-120,type=int,help="sets minimum PSD of spectrogram e.g. --psdMin=-120")
parser.add_argument("--psdMax",default=-100,type=int,help="sets maximum PSD of spectrogram e.g. --psdMax=-100")
parser.add_argument("--dur",default=120,type=int,help="sets duration of plot window e.g. --dur=120")

args = parser.parse_args()
EEG_FILENAME = args.file
psdMax = args.psdMax
psdMin = args.psdMin
nTimePoints = args.dur


CSV_FILENAME=EEG_FILENAME+'.csv'
TIMESTAMP_SIZE = 8
EEG_BUFFER_SIZE = 6336
EEG_DATA_SIZE = 227*27 #6129 bytes per buffer


#zero globals
readPos = 0
readPosEeg = 0
Eeg = []
offset = 0
lineEeg =[]
# Sampling Frequency
fs   = 1000
nFft = 8192
overlap = nFft-fs
newCount = 0
fHigh = 50
channel=3

jump = 50000
interval = 2000



sigMax = 10**(psdMax/10)
sigMin = 10**(psdMin/10)

maxNumBuffers = int(math.ceil(nTimePoints*1000/227))

with open(EEG_FILENAME, 'rb') as f:
        s=f.read(EEG_BUFFER_SIZE + TIMESTAMP_SIZE) #read in the first buffers worth
        offsetEeg = s.find(b'\x54\x69\x6D\x65') #find 'Time'
        f.close()

#set up the matplotlib figure
fig = plt.figure(figsize=(10,5))


def readEegData() : # Reads in the EEG data from the taile end of the file
    global readPosEeg,EEG_FILENAME,Eeg,TIMESTAMP_SIZE,EEG_BUFFER_SIZE,offsetEeg, EEG_DATA_SIZE, newCount,nTimePoints,maxNumBuffers
    with open(EEG_FILENAME, 'rb') as f:
        readLength = os.stat(EEG_FILENAME).st_size - offsetEeg
        readLength = readLength - (readLength % (EEG_BUFFER_SIZE+TIMESTAMP_SIZE)) #trim any incomplete data buffers 
        newDataLength = readLength - readPosEeg
        if newDataLength > (maxNumBuffers*(EEG_BUFFER_SIZE+TIMESTAMP_SIZE)) :
            readPosEeg = readLength-(maxNumBuffers*(EEG_BUFFER_SIZE+TIMESTAMP_SIZE))
            newDataLength = (maxNumBuffers*(EEG_BUFFER_SIZE+TIMESTAMP_SIZE))
        f.seek(readPosEeg+offsetEeg,0)
        #print (readPosEeg+offsetEeg)
        raw = np.fromfile(f,dtype=np.uint8, count = newDataLength)
        f.close()
        readPosEeg = readLength 
    

    #get rid of empty padding from the file
    nBuffers = newDataLength/(EEG_BUFFER_SIZE+TIMESTAMP_SIZE)
    junk=[]
    for i in range(int(nBuffers)):
        junk.extend(range(i*(EEG_BUFFER_SIZE+TIMESTAMP_SIZE),i*(EEG_BUFFER_SIZE+TIMESTAMP_SIZE)+TIMESTAMP_SIZE))
        junk.extend(range(i*(EEG_BUFFER_SIZE+TIMESTAMP_SIZE)+TIMESTAMP_SIZE+EEG_DATA_SIZE,(i+1)*(EEG_BUFFER_SIZE+TIMESTAMP_SIZE)))
    raw = np.delete(raw,junk)
    
    
     #convert the 24bit binary words into actual values
    raw = raw.astype('uint32')
    eegWords = np.squeeze(pow(2,8)*raw[2::3] + pow(2,16)*raw[1::3] + pow(2,24)*raw[0::3])
    eegWords = eegWords.astype('int32')
    eegWords = eegWords.astype('float64') *(2.4/(12*pow(2,24))) / pow(2,8)
    newCount = newCount + len(eegWords)/9
    for i in range(int(len(eegWords)/9)) :
        Eeg.append(eegWords[i*9+1:i*9+9]) # should this be +8??
    







readEegData()
newCount = 0
#print('EEG shape ' + str(len(Eeg)) + ' ' + str(len(Eeg[0])))
allVals = np.asarray(Eeg).flatten()

print('initial vals length' + str(len(allVals)))

#the following line selects which channel of data to read 
vals = allVals[channel::8]

#save the data to a csv file
np.savetxt(CSV_FILENAME, vals, delimiter=",")

#plot it
arr = plt.mlab.specgram(vals, Fs=fs,NFFT=nFft,noverlap=overlap)[0]
print('array shape ' + str(arr.shape[0]) + ' ' + str(arr.shape[1]))
im = plt.imshow(arr[:,-nTimePoints:], animated=True,origin='lower', extent=[0,nTimePoints, 0,fs/2], norm=LogNorm(vmin=sigMin, vmax=sigMax, clip=True))
#im = plt.imshow(arr[:,-nTimePoints:], animated=True,origin='lower', extent=[0,nTimePoints, 0,fs/2])

ax = plt.gca()
ax.set_ylim(0,fHigh)
# fig.set_clim([-130,-115])


#set up the animate function that gets called to update the animation
def animate(i):
    global im, ax,Eeg,newCount,channel,overlap
    
    readEegData()
    if newCount > fs : 
        allVals = np.asarray(Eeg[-(newCount+overlap):]).flatten()
        # print('new vals length' + str(len(allVals)))
        recentVals = allVals[channel::8]
        
        
        arr = plt.mlab.specgram(recentVals, Fs=fs,NFFT=nFft,noverlap=overlap)[0]
        # print('new data shape ' + str(arr.shape[0]) + ' ' + str(arr.shape[1]))
        # print('array shape ' + str(arr[0:fHigh,:].shape[0]) + ' ' + str(arr[0:fHigh,:].shape[1]))
        
        im_data = im.get_array()
        # print('image shape ' + str(im_data.shape[0]) + ' ' + str(im_data.shape[1]))
        keep_block = nTimePoints 
        im_data = np.hstack((im_data,arr))
        im_data = np.delete(im_data,np.s_[:-keep_block],1)
        

        im.set_data(im_data)
        newCount =0

    return im,

#initiate the animation 
ani = animation.FuncAnimation(fig, animate, interval=interval, blit=True)

#plt.ylim((0, 10))
plt.show()