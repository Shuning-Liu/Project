import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.stats import skew
import pyeeg 
#from scipy.fftpack import fft
import time
from scipy import signal
#import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from scipy import integrate



data = loadmat('sleep_trials_eeg_from_takashi/S1_filtered_EarEEG.mat')
X = data['cleaned_input']
epoch_length = int(data['epochLength'])
acceptEpoch = data['acceptedEpoch']

# divide data into 89 30s epochs
n = np.size(acceptEpoch)  # number of epochs
epoch = []
for ii in range(n):
    epoch.append(X[ii*epoch_length:(ii+1)*epoch_length,:])
 
'''   
############ extract feature ######################################
'''
def Hjorth(a):
    #morbidity,complexity  = pyeeg.hjorth(a)
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
                 'Gamma': (30, 100)}
    bandpower = dict()   
    for band in eeg_bands:  
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[1]        
        bandpower[band] = sum(fft_vals[freq_ix]**2)
    return bandpower

def AR(a,p):
    mod = ARMA(a, order=(p,0))
    result = mod.fit()
    parameter = result.params
    return parameter


####################### Time domain #######################################
    
# variance
variance = np.var(epoch,1)
# kurtosis
K = kurtosis(epoch,1)
# skew
S = skew(epoch,1)

# Hjorth parameters + Number of zero crossings + DFA
activity = np.zeros((n,4))
morbidity = np.zeros((n,4))
complexity = np.zeros((n,4))
zc = np.zeros((n,4))
d_f_a = np.zeros((n,4))
for ii in range(4):
    for k in range(n):
        a = epoch[k][:,ii]
        # Hjorth parameters
        activity[k,ii],morbidity[k,ii],complexity[k,ii] = Hjorth(a)
        # zero crossings
        zc[k,ii] = ((a[:-1] * a[1:]) < 0).sum()
        # DFA
        d_f_a[k,ii] = pyeeg.dfa(a)


####################### Frequency domain  ################################
'''
# periodogram method
# Define window length (4 seconds)
a = epoch[0][:,0]
win = len(a)
freqs, psd = signal.welch(a, fs, nperseg=win)
# Define delta lower and upper limits
low, high = 0, 4
# Find intersecting values in frequency vector
idx_delta = np.logical_and(freqs >= low, freqs <= high)
'''

####### Total spectral power for each band
##  fft method    t = 0.349s  bandpower[eopch][channel]['band']
fs = data['fs']
bandpower = []
for row in range(n):
    bandpower.append([])
    for col in range(4):
        a = epoch[row][:,col]
        Power = power(a,fs)       
        bandpower[row].append(Power.copy())
        

###### 需要五个 freq band 分别得到一个feature vector/matrix 89x4?
# 5 freq band power seperal for each epoch each band
delta_power = np.zeros((n,4))
theta_power = np.zeros((n,4))
alpha_power = np.zeros((n,4))
beta_power = np.zeros((n,4))
gamma_power = np.zeros((n,4))
for ii in range(4):
    for k in range(n):
        delta_power[k,ii] = bandpower[k][ii]['Delta']
        theta_power[k,ii] = bandpower[k][ii]['Theta']
        alpha_power[k,ii] = bandpower[k][ii]['Alpha']
        beta_power[k,ii] = bandpower[k][ii]['Beta']
        gamma_power[k,ii] = bandpower[k][ii]['Gamma']




#######  AR model parameters      a feature vector
# time consuming 
start = time.time()
p = 1    # order
AR_para = []
aa = 0
for row in range(n):
    AR_para.append([])
    for column in range(4):
        aa = aa+1
        a = epoch[row][:,column]
        parameter = AR(a,p)
        AR_para[row].append(parameter)
        print(aa)
end = time.time()
t = end-start



########  Spectral edge frequency (SEF) for each band in each epoch
'''
# SEF for each band
# alpha-beta 0 - 30 Hz,
spectral_edge_freq = np.zeros((n,4))
for ii in range(n):
    for k in range(4):
        a = epoch[ii][:,k]
        fft_vals = np.absolute(np.fft.rfft(a))
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(a), 1.0/fs)
        freq_ix = np.where((fft_freq >= 0) & (fft_freq <= 30))[1]
        s = np.size(fft_freq)
        freq = fft_freq.reshape(s)
        freq = freq[freq_ix]
        fft_power = (fft_vals[freq_ix])**2
        y_int = integrate.cumtrapz(fft_power, freq, initial=0)
        spectral_edge_freq[ii,k] = np.interp(max(y_int)*0.5, y_int, freq)
'''

# 可以定义def 变量 Freq 范围，power百分比
def SEF(a,freq1,freq2,r):
    fft_vals = np.absolute(np.fft.rfft(a))
    fft_freq = np.fft.rfftfreq(len(a), 1.0/fs)
    freq_ix = np.where((fft_freq >= freq1) & (fft_freq < freq2))[1]
    s = np.size(fft_freq)
    freq = fft_freq.reshape(s)
    freq = freq[freq_ix]
    fft_power = (fft_vals[freq_ix])**2
    y_int = integrate.cumtrapz(fft_power, freq, initial=0)
    spectral_edge_freq = np.interp(max(y_int)*r, y_int, freq)
    return spectral_edge_freq

sef50_delta_beta = np.zeros((n,4))  # delta-beta 0 - 30 Hz
sef50_delta_alpha = np.zeros((n,4)) # 0-12Hz
sef50_alpha = np.zeros((n,4))  # 8 - 12Hz
sef50_beta = np.zeros((n,4))  # 12 - 30Hz
for ii in range(n):
    for k in range(4):
        a = epoch[ii][:,k]
        sef50_delta_beta[ii,k] = SEF(a,0,30,0.5)   #SEF50
        sef50_delta_alpha[ii,k] = SEF(a,0,12,0.5)   #SEF50
        sef50_alpha[ii,k] = SEF(a,8,12,0.5)
        sef50_beta[ii,k] = SEF(a,12,30,0.5)
        



