import numpy as np
from scipy.io import loadmat,savemat
from scipy.stats import kurtosis
from scipy.stats import skew
import pyeeg 
import time
from scipy import signal
#import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.feature_selection import  f_classif


data1_ear = loadmat('sleep_trials_eeg_from_takashi/S1_filtered_EarEEG.mat')
data1_scalp = loadmat('sleep_trials_eeg_from_takashi/S1_filtered_ScalpEEG.mat')
data2_ear = loadmat('sleep_trials_eeg_from_takashi/S2_filtered_EarEEG.mat')
data2_scalp = loadmat('sleep_trials_eeg_from_takashi/S2_filtered_ScalpEEG.mat')
data3_ear = loadmat('sleep_trials_eeg_from_takashi/S3_filtered_EarEEG.mat')
data3_scalp = loadmat('sleep_trials_eeg_from_takashi/S3_filtered_ScalpEEG.mat')
data4_ear = loadmat('sleep_trials_eeg_from_takashi/S4_filtered_EarEEG.mat')
data4_scalp = loadmat('sleep_trials_eeg_from_takashi/S4_filtered_ScalpEEG.mat')
feature_data = loadmat('feature_data.mat')


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
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq < eeg_bands[band][1]))[1]        
        bandpower[band] = sum(fft_vals[freq_ix]**2)
        #bandpower[band] = sum(fft_vals[freq_ix]**2)/sum(fft_vals**2)  # normalised
    return bandpower

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


ear = [data1_ear,data2_ear,data3_ear,data4_ear]
scalp = [data1_scalp,data2_scalp,data3_scalp,data4_scalp]
epoch_length = int(data1_ear['epochLength'])
fs = data1_ear['fs']
'''
label3 = data3_scalp['hypnogram']
accept = data3_ear['acceptedEpoch']
accept = accept-1
label3 = label3[accept].reshape(63,1)
label = np.vstack((scalp[0]['hypnogram'][1:90],scalp[1]['hypnogram'][1:90],label3,scalp[3]['hypnogram'][1:90]))
'''
data_total = np.vstack((data1_ear['cleaned_input'],data2_ear['cleaned_input'],data3_ear['cleaned_input'],data4_ear['cleaned_input']))

#  overlapping epochs (4 files separately)
epoch_overlap = []
for n in range(4):
    eegdata = ear[n]['cleaned_input']
    for ii in range(int((np.shape(eegdata)[0]-6000)/2000 +1)):
        epoch_overlap.append(eegdata[ii*2000:ii*2000+6000,:])

'''
############################  OVERLAP FEATURE  ####################################
'''
n = len(epoch_overlap)
####################### Time ############################3333
# kurtosis
K = kurtosis(epoch_overlap,1)
# skew
S = skew(epoch_overlap,1)

activity = np.zeros((n,4))
morbidity = np.zeros((n,4))
complexity = np.zeros((n,4))
zc = np.zeros((n,4))
d_f_a = np.zeros((n,4))
for ii in range(4):
    for k in range(n):
        a = epoch_overlap[k][:,ii]
        # Hjorth parameters
        activity[k,ii],morbidity[k,ii],complexity[k,ii] = Hjorth(a)
        # zero crossings
        zc[k,ii] = ((a[:-1] * a[1:]) < 0).sum()
        # DFA
        d_f_a[k,ii] = pyeeg.dfa(a)
feature_matrix_time = np.hstack((K,S,activity,morbidity,complexity,zc,d_f_a))

####################    Freq ###############################
bandpower = []
for row in range(n):
    bandpower.append([])
    for col in range(4):
        a = epoch_overlap[row][:,col]
        Power = power(a,fs)
        bandpower[row].append(Power.copy())
# normalised band power
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

# SEF
sef50_delta_beta = np.zeros((n,4))  # delta-beta 0 - 30 Hz
sef95_delta_beta = np.zeros((n,4))  # delta-beta 0 - 30 Hz
sefd_delta_beta = np.zeros((n,4))  # SEF95-SEF50
for ii in range(n):
    for k in range(4):
        a = epoch_overlap[ii][:,k]           
        sef50_delta_beta[ii,k] = SEF(a,0,30,0.5)   #SEF50                       
        sef95_delta_beta[ii,k] = SEF(a,0,30,0.95)  #SEF95                                
sefd_delta_beta = sef95_delta_beta - sef50_delta_beta # SEFd

feature_matrix_freq = np.hstack((delta_power,theta_power,alpha_power,beta_power,gamma_power,sef50_delta_beta,sef95_delta_beta,sefd_delta_beta))
# total feature matrix (time + freq)
feature_matrix_overlap = np.hstack((feature_matrix_time,feature_matrix_freq))

selected_idx_total = feature_data['select_idx_total'].ravel()
feature_overlap_selected = feature_matrix_overlap[:,selected_idx_total.astype(int)]

feature_overlap_data = {'feature_overlap_selected': feature_overlap_selected,'feature_total':feature_matrix_overlap}
savemat('feature_overlap_data.mat',feature_overlap_data)





