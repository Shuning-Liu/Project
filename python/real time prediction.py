import pandas as pd
import time
import numpy as np
from scipy.io import loadmat,savemat
#from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.stats import kurtosis
from scipy.stats import skew
import pyeeg 
from scipy import signal
from scipy import integrate
import warnings


start = time.time()
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

warnings.filterwarnings('ignore')

'''
############################## SVM classifer ###############################
'''
#################### train data ###################################
feature_data = loadmat('feature_data.mat')
label = feature_data['label'].ravel()
fs = 200
feature_matrix_selected = feature_data['feature_selected']
select_idx_total = feature_data['select_idx_total'].ravel()
Xtrain = feature_matrix_selected
svclassifier = SVC(gamma=0.000005,C = 200, kernel='rbf',decision_function_shape='ovr')
svclassifier.fit(Xtrain, label)
end = time.time()
t = end - start


count = 0
aa = []
while True:
    time.sleep(29)
    filename = r'eeg.csv'
    data = pd.read_csv(filename,header=None,skiprows=count*6000+1,nrows=6000)
    aa.append(data)    
    count = count+1
    print(count)
    
    start = time.time()
    ####################### calculate features #####################
    epoch = data.to_numpy()   
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
    print(y_pred)
    end = time.time()
    t1 = end - start
    '''
    if count == 3:
        break
    '''
    













