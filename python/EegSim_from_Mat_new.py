import numpy as np
from scipy.io import loadmat
from scipy import signal
from fractions import gcd
from math import ceil
import time
import struct
import argparse
import csv
import os

###########################
#Read in Arguments & set default parameters
###########################
parser = argparse.ArgumentParser(description='Simulate realtime Hearables data from a .mat file')
parser.add_argument("--ipFile",default='sleep_trials_eeg_from_takashi/S1_filtered_EarEEG.mat',help="--ipFile='path_to_file'")

parser.add_argument("--opFile",default='eeg.csv',help="--opFile='path_to_file'")

args = parser.parse_args()


UPDATE_PERIOD = 1 #update rate in seconds

#############################
#Load the data from mat file
#############################

x = loadmat(args.ipFile )
y = loadmat(args.ipFile.replace('EarEEG','ScalpEEG') )
ipData = x['input_data']  #don't change this to cleaned data
fs = x['fs']
hypnogram = y['hypnogram']
dataRows,dataColumns = ipData.shape
batch_size = int(UPDATE_PERIOD*fs)

epochLength = x['epochLength'] 



#############################
#Write the data in a loop to a file, with approximate timing and timestamp incrementing
#############################

tStart = time.time()
updateCount = 1
loopCount = 0
currentEpoch = 0

with open(args.opFile, 'w',newline='') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    while True:
        loopCount +=1
        for i in range(dataRows) :            
            if (i+1) % batch_size == 0:
                print (time.time())
                while time.time() < tStart + updateCount * UPDATE_PERIOD:
                    time.sleep(0.01)
                updateCount += 1
            #if (i+1) % epochLength ==0: currentEpoch+=1
            
            #csvfile.write(str(hypnogram[currentEpoch])[1:-1] +',')
            #csvwriter.writerow(ipData[i])
            aa = np.hstack((hypnogram[currentEpoch],ipData[i]))
            csvwriter.writerow(aa)
            if (i+1) % epochLength ==0: currentEpoch+=1
        if loopCount > 10 : break
        

'''
        
     
def readPpgData(): 
    global readPos, temp_red, temp_ir, temp_green, args  #get global variables
    
    with open(args.ipFile, 'r') as f:     #open file
        csvreader = csv.reader(f, delimiter=',', quotechar='|')   #tell python how to interpret the csv file
        f.seek(readPos,os.SEEK_SET) #seek to read position 
        for row in csvreader: #read all the rows
            if len(row)==3:   #needs to be modified
                
                new_red = float(row[0])
                new_ir = float(row[1])
                new_green = float(row[2])             
                
                
                #Temp_red is a rolling buffer, e.g.  temp_red = np.zeros(BUFF_SIZE,dtype=float)
                #This code rolls all the data by 1 sample and then appends the latest sample on the end
                #You may want to use something other than a rolling buffer. 
                
                temp_red = np.roll(temp_red,-1)
                temp_ir = np.roll(temp_ir,-1)
                temp_green = np.roll(temp_green,-1)
                
                temp_red[-1] = new_red
                temp_ir[-1] = new_ir
                temp_green[-1] = new_green
            
        readPos = f.tell() #Get the current read position of the file
        
'''