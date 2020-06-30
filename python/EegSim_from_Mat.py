# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:07:13 2020

@author: Shuning Liu
"""

import numpy as np
from scipy.io import loadmat
from scipy import signal
from fractions import gcd
from math import ceil
import time
import struct
import argparse


###########################
#Read in Arguments & set default parameters
###########################
parser = argparse.ArgumentParser(description='Simulate realtime Hearables data from a .mat file')
parser.add_argument("--ipFile",default='sleep_trials_eeg_from_takashi/S1_filtered_EarEEG.mat',help="--ipFile='path_to_file'")

parser.add_argument("--opFile",default='eeg.bin',help="--opFile='path_to_file'")

args = parser.parse_args()


EEG_BUFFER_SIZE = 6336
EEG_BUFFER_SAMPLES = 227
EEG_DATA_SIZE = 227*27 #6129 bytes per buffer
TIMESTAMP_SIZE = 8
OP_FS = 1000
TIMESTAMP_CLK_FREQ = 31250

KEYWORD = b'\x54\x69\x6D\x65' #Time
TIMESTAMP_STEP = int(ceil(TIMESTAMP_CLK_FREQ * EEG_BUFFER_SAMPLES / OP_FS))

#############################
#Load the data from mat file
#############################

x = loadmat(args.ipFile )
ipData = x['input_data']
fs = x['fs']


#############################
#trim to a complete block size (blocks are 227 samples)
#############################
rows,columns = ipData.shape
validRows = rows - (rows%EEG_BUFFER_SAMPLES)
ipData = np.delete(ipData, np.s_[validRows::],0)

#############################
#resample it to 1kHz
#############################
divisor = gcd(OP_FS,fs) #get the greatest common divisor
print(divisor)
raw_data = signal.resample_poly(ipData,OP_FS/divisor, fs/divisor) #resamples along axis 0 by default


#############################
#pad/clip the data to 8 channels + 1 channel of filler because the ads129x has 8 channels of data and 1 channel of configuration info
#############################
rows, columns = raw_data.shape
print(raw_data.shape)
if columns > 8:   #too many columns
    np.delete(raw_data, np.s_[8::],1) #clip extra columns
else :
    #pad by cloning the whole array
    while (8 - columns)>= columns:
        raw_data = np.hstack((raw_data,raw_data))
        row, columns = raw_data.shape
        
    #if there are still columns left to fill then pad individually
    if columns <8:
        raw_data = np.hstack((raw_data,raw_data[:,:8-columns]))

raw_data = np.pad(raw_data,((0,0),(1,0)),'constant',constant_values=0)

print(raw_data.shape)




#############################
#flatten it and convert it to binary. the first multiplier is related to the ADC and the second is just to shift it by 8 bits
#############################
dataStream = (raw_data.flatten()  * (12*pow(2,24))/2.4) * pow(2,8)
dataStream = dataStream.astype('int32')





#############################
#pad it to mimic data from the device & write it to file
#############################
firstByte = dataStream >>24
secondByte = (np.bitwise_and(dataStream,0x00ff0000)) >>16 
thirdByte = (np.bitwise_and(dataStream,0x0000ff00)) >>8

bytes = np.concatenate([[firstByte.astype('uint8')],[secondByte.astype('uint8')],[thirdByte.astype('uint8')]])
byteStream = bytes.flatten('F')

dataBlocks = np.reshape(byteStream,(-1,EEG_DATA_SIZE))



#############################
#Write the data in a loop to a file, with approximate timing and timestamp incrementing
#############################

timestamp = int(0)
padding = np.zeros (EEG_BUFFER_SIZE-EEG_DATA_SIZE,dtype=np.uint8)
tStart = time.time()

with open(args.opFile, 'wb') as f:
    while True :
        tLoopStart = time.time()
        for i in range(len(dataBlocks)): #len returns num of rows by default
            f.write(KEYWORD) #write keyword "Time"
            f.write(struct.pack(">I",timestamp)) #convert integer to uint32 & write 
            f.write(dataBlocks[i,:]) #write data
            f.write(padding) #pad the rest of the block with zeros
            f.flush() #flush data so other scripts can access it
            while (time.time() - tLoopStart) < i +1:
                time.sleep(0.1)
            timestamp = timestamp + TIMESTAMP_STEP
            print("looping time elapsed:" + str(time.time()-tStart ))
            
f.close()