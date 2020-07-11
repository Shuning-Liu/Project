import mne
import numpy as np
from scipy.io import loadmat,savemat
from scipy.stats import kurtosis
from scipy.stats import skew
import pyeeg 
from scipy import signal
#import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.feature_selection import  f_classif

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

def two_state_fscore(a,b):
    idx_ab = np.where((label == a) | (label == b))[0]
    label_ab = label[idx_ab]
    feature_ab = feature_matrix[idx_ab,:]
    f,p = f_classif(feature_ab,label_ab)
    f_ab = np.zeros(int(len(f)/10))
    for ii in range(int(len(f)/10)):
        f_ab[ii] = max(f[ii*10:(ii+1)*10])
    return f_ab


data = mne.io.read_raw_edf('raw data/n1.edf')
raw_data = data.get_data()
info = data.info

eeg = loadmat('hypn1.mat')
hyp = eeg['hyp']
fs = 512
data = np.transpose(raw_data)
n = np.shape(hyp)[0]
epoch_length = 30 * fs
data = data[:n*epoch_length,1:11]

epoch = []
for ii in range(n):
    epoch.append(data[ii*epoch_length:(ii+1)*epoch_length,:])
    
'''
############################  FEATURE  ####################################
'''
####################### Time ############################3333

# kurtosis
K = kurtosis(epoch,1)
# skew
S = skew(epoch,1)


activity = np.zeros((n,10))
morbidity = np.zeros((n,10))
complexity = np.zeros((n,10))
zc = np.zeros((n,10))
#d_f_a = np.zeros((n,10))
for ii in range(10):
    for k in range(n):
        a = epoch[k][:,ii]
        # Hjorth parameters
        activity[k,ii],morbidity[k,ii],complexity[k,ii] = Hjorth(a)
        # zero crossings
        zc[k,ii] = ((a[:-1] * a[1:]) < 0).sum()
        #d_f_a[k,ii] = pyeeg.dfa(a)

feature_matrix_time = np.hstack((K,S,activity,morbidity,complexity,zc))

####################    Freq ###############################

bandpower = []
for row in range(n):
    bandpower.append([])
    for col in range(10):
        a = epoch[row][:,col]
        Power = power(a,fs)
        bandpower[row].append(Power.copy())

# normalised band power
delta_power = np.zeros((n,10))
theta_power = np.zeros((n,10))
alpha_power = np.zeros((n,10))
beta_power = np.zeros((n,10))
gamma_power = np.zeros((n,10))
for ii in range(10):
    for k in range(n):
        delta_power[k,ii] = bandpower[k][ii]['Delta']
        theta_power[k,ii] = bandpower[k][ii]['Theta']
        alpha_power[k,ii] = bandpower[k][ii]['Alpha']
        beta_power[k,ii] = bandpower[k][ii]['Beta']
        gamma_power[k,ii] = bandpower[k][ii]['Gamma']

# SEF
sef50_delta_beta = np.zeros((n,10))  # delta-beta 0 - 30 Hz
sef95_delta_beta = np.zeros((n,10))  # delta-beta 0 - 30 Hz
sefd_delta_beta = np.zeros((n,10))  # SEF95-SEF50
for ii in range(n):
    for k in range(10):
        a = epoch[ii][:,k]
        sef50_delta_beta[ii,k] = SEF(a,0,30,0.5)   #SEF50                       
        sef95_delta_beta[ii,k] = SEF(a,0,30,0.95)  #SEF95                                
sefd_delta_beta = sef95_delta_beta - sef50_delta_beta # SEFd


feature_matrix_freq = np.hstack((delta_power,theta_power,alpha_power,beta_power,gamma_power,sef50_delta_beta,sef95_delta_beta,sefd_delta_beta))
# total feature matrix (time + freq)
feature_matrix = np.hstack((feature_matrix_time,feature_matrix_freq))


##################### f scores for two states #################
label = hyp[:,0]

f_01 = two_state_fscore(0,1)   # state W vs N1
f_02 = two_state_fscore(0,2)   # state W vs N2
f_03 = two_state_fscore(0,3)   # state W vs N3
f_04 = two_state_fscore(0,4)
f_05 = two_state_fscore(0,5)
f_12 = two_state_fscore(1,2)   # state N1 vs N2
f_13 = two_state_fscore(1,3)   # state N1 vs N3
f_14 = two_state_fscore(1,4)
f_15 = two_state_fscore(1,5)
f_23 = two_state_fscore(2,3)   # state N2 vs N3
f_24 = two_state_fscore(2,4)
f_25 = two_state_fscore(2,5)
f_34 = two_state_fscore(3,4)
f_35 = two_state_fscore(3,5)
f_45 = two_state_fscore(4,5)


f_two = np.vstack((f_01,f_02,f_03,f_04,f_05,f_12,f_13,f_14,f_15,f_23,f_24,f_25,f_34,f_35,f_45))
idx = np.argsort(-f_two,axis=1)
idx = idx[:,0:10]
count = np.zeros((1,14))
for ii in range(14):
    count[0,ii] = np.sum(idx == ii)

idx_select = np.where(count >= 13)[1]

idx_total = np.zeros((1,len(idx_select)*10))
for m in range(int(len(idx_select))):
    idx_total[0,m*10:(m+1)*10] = np.array([[idx_select[m]*10,idx_select[m]*10+1,idx_select[m]*10+2,idx_select[m]*10+3,idx_select[m]*10+4,idx_select[m]*10+5,idx_select[m]*10+6,idx_select[m]*10+7,idx_select[m]*10+8,idx_select[m]*10+9]])
idx_total = idx_total.ravel()

feature_matrix_selected = feature_matrix[:,idx_total.astype(int)]

# transition matrix using total scalp EEG hyponogram
y_cut = label[0:len(label)-1]
N_next_state = np.zeros((6,6))
for pre in range(6):
    num1 = np.where(y_cut == pre)[0]
    next_state = label[num1+1]
    for ii in range(6):
        N_next_state[pre,ii] = N_next_state[pre,ii] + sum(next_state == ii)

transition_matrix = np.zeros((6,6))
for ii in range(6):
    for k in range(6):
        transition_matrix[ii,k] = N_next_state[ii,k]/sum(N_next_state[ii,:])

feature_newdata = {'feature_selected': feature_matrix_selected,'feature_total':feature_matrix ,'label': label, 'transition_matrix': transition_matrix, 'EEGdata': data,'selected_idx':idx_select,'select_idx_total':idx_total}
savemat('feature_new_data.mat',feature_newdata)




