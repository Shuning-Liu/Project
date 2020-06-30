import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.stats import skew
import pyeeg 
import time
from scipy import signal
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from scipy import integrate
from sklearn.feature_selection import  SelectKBest,f_classif,f_regression

data1_ear = loadmat('sleep_trials_eeg_from_takashi/S1_filtered_EarEEG.mat')
data1_scalp = loadmat('sleep_trials_eeg_from_takashi/S1_filtered_ScalpEEG.mat')
data2_ear = loadmat('sleep_trials_eeg_from_takashi/S2_filtered_EarEEG.mat')
data2_scalp = loadmat('sleep_trials_eeg_from_takashi/S2_filtered_ScalpEEG.mat')
data3_ear = loadmat('sleep_trials_eeg_from_takashi/S3_filtered_EarEEG.mat')
data3_scalp = loadmat('sleep_trials_eeg_from_takashi/S3_filtered_ScalpEEG.mat')
data4_ear = loadmat('sleep_trials_eeg_from_takashi/S4_filtered_EarEEG.mat')
data4_scalp = loadmat('sleep_trials_eeg_from_takashi/S4_filtered_ScalpEEG.mat')

# try the first file
X = data2_ear['cleaned_input']
epoch_length = int(data2_ear['epochLength'])
acceptEpoch = data2_ear['acceptedEpoch']
label = data2_scalp['hypnogram']
label = label[1:90]

# divide data into 89 30s epochs
n = np.size(acceptEpoch)  # number of epochs
epoch = []
for ii in range(n):
    epoch.append(X[ii*epoch_length:(ii+1)*epoch_length,:])

'''   
#################### Feature Extraction ######################################
'''
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


####################### Time domain #######################################
t1 = time.time()
# kurtosis
K = kurtosis(epoch,1)
t2 = time.time()
# skew
S = skew(epoch,1)
t3 = time.time()
t_K = t2 - t1
t_S = t3 - t2

# Hjorth parameters + Number of zero crossings + DFA
activity = np.zeros((n,4))
morbidity = np.zeros((n,4))
complexity = np.zeros((n,4))
zc = np.zeros((n,4))
d_f_a = np.zeros((n,4))
t_Hjorth = 0
t_zc = 0
t_dfa = 0
for ii in range(4):
    for k in range(n):
        a = epoch[k][:,ii]
        # Hjorth parameters
        t1 = time.time()
        activity[k,ii],morbidity[k,ii],complexity[k,ii] = Hjorth(a)
        t2 = time.time()
        t_Hjorth = t_Hjorth + t2-t1
        # zero crossings
        zc[k,ii] = ((a[:-1] * a[1:]) < 0).sum()
        t3 = time.time()
        t_zc = t_zc + t3-t2
        # DFA
        d_f_a[k,ii] = pyeeg.dfa(a)
        t4 = time.time()
        t_dfa = t_dfa + t4 - t3

feature_matrix_time = np.hstack((K,S,activity,morbidity,complexity,zc,d_f_a))

print(t_K)
print(t_S)
print(t_Hjorth)
print(t_zc)
print(t_dfa)

####################### Frequency domain  ################################
####### Total spectral power for each band
##  fft method    t = 0.349s  bandpower[eopch][channel]['band']
fs = data1_ear['fs']
bandpower = []
start = time.time()
for row in range(n):
    bandpower.append([])
    for col in range(4):
        a = epoch[row][:,col]
        Power = power(a,fs)       
        bandpower[row].append(Power.copy())   
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
end = time.time()
t_freqpower = end - start

########  Spectral edge frequency (SEF) for each band in each epoch
# 变量 Freq 范围，power百分比
sef50_delta_beta = np.zeros((n,4))  # delta-beta 0 - 30 Hz
sef95_delta_beta = np.zeros((n,4))  # delta-beta 0 - 30 Hz
sefd_delta_beta = np.zeros((n,4))  # SEF95-SEF50
t_SEF50 = 0
t_SEF95 = 0
for ii in range(n):
    for k in range(4):
        a = epoch[ii][:,k]
        t1 = time.time()
        sef50_delta_beta[ii,k] = SEF(a,0,30,0.5)   #SEF50
        t2 = time.time()
        t_SEF50 = t_SEF50 + t2-t1
        sef95_delta_beta[ii,k] = SEF(a,0,30,0.95)  #SEF95        
        t3 = time.time()
        t_SEF95 = t_SEF95 + t2-t1

sefd_delta_beta = sef95_delta_beta - sef50_delta_beta # SEFd
t_SEFd = t_SEF95 + t_SEF50

feature_matrix_freq = np.hstack((delta_power,theta_power,alpha_power,beta_power,gamma_power,sef50_delta_beta,sef95_delta_beta,sefd_delta_beta))

# total feature matrix (time + freq)
feature_matrix = np.hstack((feature_matrix_time,feature_matrix_freq))
print(t_freqpower)
print(t_SEF50)
print(t_SEF95)
print(t_SEFd)

'''
######################### f-test/ feature selection ##########################
'''
# f scores for each feature each channel
f,p = f_classif(feature_matrix,label.ravel())

# choose the highest f score among 4 channels for each feature
f_score = np.zeros(int(len(f)/4))
for ii in range(int(len(f)/4)):
    f_score[ii] = max(f[ii*4:(ii+1)*4])
    

############## plot time vs score
t = np.array([t_S,t_K,t_Hjorth,t_Hjorth,t_Hjorth,t_zc,t_dfa,t_freqpower,t_freqpower,t_freqpower,t_freqpower,t_freqpower,t_SEF50,t_SEF95,t_SEFd])

fig,ax=plt.subplots()
ax.scatter(t,f_score,c='r')
#plt.scatter(t,f_score)
n = np.arange(15)
for i,txt in enumerate(n):
    ax.annotate(txt,(t[i],f_score[i]))
plt.ylabel('f score')
plt.xlabel('time')




