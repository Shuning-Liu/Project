import numpy as np
import csv
import os
import time
from scipy.io import loadmat,savemat
#from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.stats import kurtosis
from scipy.stats import skew
import pyeeg 
from scipy import signal
from scipy import integrate
import warnings

class CircularBuffer(object):
    def __init__(self, max_size=10):
        """Initialize the CircularBuffer with a max_size if set, otherwise
        max_size will elementsdefault to 10"""
        self.buffer = [None] * max_size
        self.head = 0
        self.tail = 0
        self.max_size = max_size
        
    def __str__(self):
        """Return a formatted string representation of this CircularBuffer."""
        items = ['{!r}'.format(item) for item in self.buffer]
        return '[' + ', '.join(items) + ']'
    
    def size(self):
        """Return the size of the CircularBuffer
        Runtime: O(1) Space: O(1)"""
        if self.tail >= self.head:
            return self.tail - self.head
        if self.tail < self.head:
            return self.max_size - self.head + self.tail
        #return self.max_size - self.head - self.tail
    
    def is_empty(self):
        """Return True if the head of the CircularBuffer is equal to the tail,
        otherwise return False
        Runtime: O(1) Space: O(1)"""
        return self.tail == self.head
    
    def is_full(self):
        """Return True if the tail of the CircularBuffer is one before the head,
        otherwise return False
        Runtime: O(1) Space: O(1)"""
        return self.tail == (self.head-1) % self.max_size

    def enqueue(self, item):
        """Insert an item at the back of the CircularBuffer
        Runtime: O(1) Space: O(1)"""
        
        if self.is_full():
            raise OverflowError(
                "CircularBuffer is full, unable to enqueue item")
        
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.max_size

    def front(self):
        """Return the item at the front of the CircularBuffer
        Runtime: O(1) Space: O(1)"""
        return self.buffer[self.head]

    def dequeue(self):
        """Return the item at the front of the Circular Buffer and remove it
        Runtime: O(1) Space: O(1)"""
        if self.is_empty():
            raise IndexError("CircularBuffer is empty, unable to dequeue")
        item = self.buffer[self.head]
        self.buffer[self.head] = None
        self.head = (self.head + 1) % self.max_size
        return item

'''
###########################################################################
'''
warnings.filterwarnings('ignore')

def Hjorth(a):
    first_deriv = np.diff(a)
    second_deriv = np.diff(a,2)
    var_zero = np.var(a)   # varianve of signal
    var_d1 = np.var(first_deriv)   # variance of first derivative
    var_d2 = np.var(second_deriv)  # variance of second derivative
    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity
    return activity,morbidity,complexity


def power(a,fs):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(a))
    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(a), 1.0/fs)
    # Define EEG bands
    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}
    bandpower = dict()   
    for band in eeg_bands:  
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq < eeg_bands[band][1]))        
        bandpower[band] = sum(fft_vals[freq_ix]**2)
        #bandpower[band] = sum(fft_vals[freq_ix]**2)/sum(fft_vals**2)  # normalised
    return bandpower

def SEF(a,freq1,freq2,r):
    fft_vals = np.absolute(np.fft.rfft(a))
    fft_freq = np.fft.rfftfreq(len(a), 1.0/fs)
    freq_ix = np.where((fft_freq >= freq1) & (fft_freq < freq2))
    s = np.size(fft_freq)
    freq = fft_freq.reshape(s)
    freq = freq[freq_ix]
    fft_power = (fft_vals[freq_ix])**2
    y_int = integrate.cumtrapz(fft_power, freq, initial=0)
    spectral_edge_freq = np.interp(max(y_int)*r, y_int, freq)
    return spectral_edge_freq

def read_data(buffer,readPos):
    f = open('eeg.csv', 'r')
    csvreader = csv.reader(f, delimiter=',', quotechar='|')
    f.seek(readPos,os.SEEK_SET)
    data = [row for row in csvreader]
    if len(data) > epochsize*4:
        a = len(data) - epochsize
        a = a - (a % epochsize)     
        data = data[a:]      #  从6000的整数倍开始
    for ii in range(int(len(data))):
        buffer.enqueue(data[ii])
    #print(len(data))
    readPos = f.tell()
    f.close()
    return buffer,readPos


'''
############################## SVM classifer ###############################
'''
#################### train data ###################################
feature_data = loadmat('feature_data.mat')
label = feature_data['label'].ravel()
fs = 200
feature_matrix_selected = feature_data['feature_selected']
transition_matrix = feature_data['transition_matrix']
select_idx_total = feature_data['select_idx_total'].ravel()
Xtrain = feature_matrix_selected
svclassifier = SVC(gamma=0.000005,C = 200, kernel='rbf',decision_function_shape='ovr')
svclassifier.fit(Xtrain, label)

readPos = 0
epochsize = 6000
buffer = CircularBuffer(epochsize*4)
predict = []
count = 0
while True:
    buffer,readPos = read_data(buffer,readPos)
    if buffer.size() >= epochsize:
        pop = []
        for ii in range(epochsize):
            pop.append(buffer.dequeue())
        #print(count)
        #######################  calculate features #####################
        
        data_epoch = []
        for ii in range(epochsize):
            data_epoch.append(np.array(pop[ii],dtype=float))
        data_epoch = np.array(data_epoch)   # 到这个位置开始提取 data & hypnogram
        
        
        epoch = data_epoch[:,1:]
        hypnogram = data_epoch[5,0]      # take the fifth hypnogram as the true state
        
        K = kurtosis(epoch)
        S = skew(epoch)
        activity = np.zeros(4)
        morbidity = np.zeros(4)
        complexity = np.zeros(4)
        zc = np.zeros(4)
        d_f_a = np.zeros(4)
        for k in range(4):
            activity[k],morbidity[k],complexity[k] = Hjorth(epoch[:,k])
            zc[k] = ((epoch[:,k][:-1] * epoch[:,k][1:]) < 0).sum()
            d_f_a[k] = pyeeg.dfa(epoch[:,k])  
        feature_time = np.hstack((K,S,activity,morbidity,complexity,zc,d_f_a))
        
        ############## freq ################################
        bandpower = []
        for col in range(4):
            Power = power(epoch[:,col],fs)
            bandpower.append(Power.copy())
        delta_power = np.zeros(4)
        theta_power = np.zeros(4)
        alpha_power = np.zeros(4)
        beta_power = np.zeros(4)
        gamma_power = np.zeros(4)
        for ii in range(4):  
            delta_power[ii] = bandpower[ii]['Delta']
            theta_power[ii] = bandpower[ii]['Theta']
            alpha_power[ii] = bandpower[ii]['Alpha']
            beta_power[ii] = bandpower[ii]['Beta']
            gamma_power[ii] = bandpower[ii]['Gamma']
        # SEF
        sef50_delta_beta = np.zeros(4)
        sef95_delta_beta = np.zeros(4) 
        sefd_delta_beta = np.zeros(4)
        for k in range(4):
            a = epoch[:,k]           
            sef50_delta_beta[k] = SEF(a,0,30,0.5)   #SEF50                       
            sef95_delta_beta[k] = SEF(a,0,30,0.95)  #SEF95                                
        sefd_delta_beta = sef95_delta_beta - sef50_delta_beta # SEFd
        feature_freq = np.hstack((delta_power,theta_power,alpha_power,beta_power,gamma_power,sef50_delta_beta,sef95_delta_beta,sefd_delta_beta))
        feature = np.hstack((feature_time,feature_freq))   
        feature_select = feature[select_idx_total.astype(int)]
        feature_select = feature_select.reshape(1,len(feature_select))
        
        ################ prediction ###########################
        y_pred = svclassifier.predict(feature_select)
        #print(y_pred)
        
        # probability SVC returns by decision function
        p = np.array(svclassifier.decision_function(feature_select)) # decision is a voting function
        prob = np.exp(p)/(np.sum(np.exp(p)))
        prob = prob.ravel()
        #print(y_pred == hypnogram)
        #print('The probability for the predicted state is',round(float(prob[y_pred]),3))
        
        ##################### conditional prob
        if count == 0:
            print(y_pred)
            print(y_pred == hypnogram)
            print('The probability for the predicted state is',round(float(prob[y_pred]),3))
            predict.append(y_pred)            
        else:
            label_previous = predict[count - 1]
            con_prob = np.zeros(4)
            for q in range(4):
                con_prob[q] = transition_matrix[label_previous,q] * prob[q]
            y_pred_con = np.argmax(con_prob)
            con_prob_normalised = con_prob / sum(con_prob)
            print(y_pred_con)
            print(y_pred_con == hypnogram)
            print('The probability for the predicted state is',round(float(con_prob_normalised[y_pred_con]),3))      
            predict.append(y_pred_con)
        count = count + 1
        














