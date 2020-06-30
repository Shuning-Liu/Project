import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis
from scipy.stats import skew
import pyeeg 
import time
#from scipy import signal
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.feature_selection import  f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

data1_ear = loadmat('sleep_trials_eeg_from_takashi/S1_filtered_EarEEG.mat')
data1_scalp = loadmat('sleep_trials_eeg_from_takashi/S1_filtered_ScalpEEG.mat')
data2_ear = loadmat('sleep_trials_eeg_from_takashi/S2_filtered_EarEEG.mat')
data2_scalp = loadmat('sleep_trials_eeg_from_takashi/S2_filtered_ScalpEEG.mat')
data3_ear = loadmat('sleep_trials_eeg_from_takashi/S3_filtered_EarEEG.mat')
data3_scalp = loadmat('sleep_trials_eeg_from_takashi/S3_filtered_ScalpEEG.mat')
data4_ear = loadmat('sleep_trials_eeg_from_takashi/S4_filtered_EarEEG.mat')
data4_scalp = loadmat('sleep_trials_eeg_from_takashi/S4_filtered_ScalpEEG.mat')

start = time.time()

ear = [data1_ear,data2_ear,data3_ear,data4_ear]
scalp = [data1_scalp,data2_scalp,data3_scalp,data4_scalp]

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
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq < eeg_bands[band][1]))[1]        
        #bandpower[band] = sum(fft_vals[freq_ix]**2)
        bandpower[band] = sum(fft_vals[freq_ix]**2)/sum(fft_vals**2)  # normalised
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


epoch_length = int(data1_ear['epochLength'])
fs = data1_ear['fs']
label3 = data3_scalp['hypnogram']
accept = data3_ear['acceptedEpoch']
accept = accept-1
label3 = label3[accept].reshape(63,1)
y = [scalp[0]['hypnogram'][1:90],scalp[1]['hypnogram'][1:90],label3,scalp[3]['hypnogram'][1:90]]


#feature_matrix_total = np.zeros((89*3+63,60))
# 循环
for p in range(4):
    X = ear[p]['cleaned_input']
    acceptEpoch = ear[p]['acceptedEpoch']
    n = np.size(acceptEpoch)  # number of epochs
    label = y[p]
    epoch = []
    for ii in range(n):
        epoch.append(X[ii*epoch_length:(ii+1)*epoch_length,:])
 ####################### Time domain #######################################
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
    feature_matrix_time = np.hstack((K,S,activity,morbidity,complexity,zc,d_f_a))
  ####################### Frequency domain  ################################
    bandpower = []
    for row in range(n):
        bandpower.append([])
        for col in range(4):
            a = epoch[row][:,col]
            Power = power(a,fs)       
            bandpower[row].append(Power.copy())   
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

    ########  Spectral edge frequency (SEF) for each band in each epoch
# 变量 Freq 范围，power百分比
    sef50_delta_beta = np.zeros((n,4))  # delta-beta 0 - 30 Hz
    sef95_delta_beta = np.zeros((n,4))  # delta-beta 0 - 30 Hz
    sefd_delta_beta = np.zeros((n,4))  # SEF95-SEF50
    for ii in range(n):
        for k in range(4):
            a = epoch[ii][:,k]           
            sef50_delta_beta[ii,k] = SEF(a,0,30,0.5)   #SEF50                       
            sef95_delta_beta[ii,k] = SEF(a,0,30,0.95)  #SEF95                                
    sefd_delta_beta = sef95_delta_beta - sef50_delta_beta # SEFd

    feature_matrix_freq = np.hstack((delta_power,theta_power,alpha_power,beta_power,gamma_power,sef50_delta_beta,sef95_delta_beta,sefd_delta_beta))
    # total feature matrix (time + freq)
    feature_matrix = np.hstack((feature_matrix_time,feature_matrix_freq))    
    # combine feature matrixes
    if p == 0:
        feature_matrix_total = feature_matrix
    else:
        feature_matrix_total = np.vstack((feature_matrix_total,feature_matrix))
    
        
################ f scores for two states 
label_total = np.vstack((y[0],y[1],y[2],y[3])).ravel()

def two_state_fscore(a,b):
    idx_ab = np.where((label_total == a) | (label_total == b))[0]
    label_ab = label_total[idx_ab]
    feature_ab = feature_matrix_total[idx_ab,:]
    f,p = f_classif(feature_ab,label_ab)
    f_ab = np.zeros(int(len(f)/4))
    for ii in range(int(len(f)/4)):
        f_ab[ii] = max(f[ii*4:(ii+1)*4])
    return f_ab

f_01 = two_state_fscore(0,1)   # state W vs N1
f_02 = two_state_fscore(0,2)   # state W vs N2
f_03 = two_state_fscore(0,3)   # state W vs N3
f_12 = two_state_fscore(1,2)   # state N1 vs N2
f_13 = two_state_fscore(1,3)   # state N1 vs N3
f_23 = two_state_fscore(2,3)   # state N2 vs N3


f_two = np.vstack((f_01,f_02,f_03,f_12,f_13,f_23))
idx = np.argsort(-f_two,axis=1)
idx = idx[:,0:10]
count = np.zeros((1,15))
for ii in range(15):
    count[0,ii] = np.sum(idx == ii)

idx_select = np.where(count >= 4)[1]

# Feature selection (four files four channels)
idx_total = np.zeros((1,len(idx_select)*4))
for m in range(int(len(idx_select))):
    idx_total[0,m*4:(m+1)*4] = np.array([[idx_select[m]*4,idx_select[m]*4+1,idx_select[m]*4+2,idx_select[m]*4+3]])   
idx_total = idx_total.reshape(np.size(idx_total))

feature_matrix_selected = feature_matrix_total[:,idx_total.astype(int)]


'''
#########################  SVM classification ################################
'''
# split into train test data
label_total = np.vstack((y[0],y[1],y[2],y[3]))
# random state = default  accuracy = 

number = range(len(label_total))

Xtrain, Xtest, ytrain_idx, ytest_idx = train_test_split(feature_matrix_selected, number, test_size = 0.2)
#X_train, X_test, y_train, y_test = train_test_split(feature_matrix_selected, label_total, test_size = 0.2)

ytrain = label_total[ytrain_idx]
ytest = label_total[ytest_idx]

svclassifier = SVC(gamma=0.000005,C = 200, kernel='rbf',decision_function_shape='ovo',probability=True)

svclassifier.fit(Xtrain, ytrain.ravel())
y_pred = svclassifier.predict(Xtest)

print(confusion_matrix(ytest, y_pred))
print(classification_report(ytest, y_pred))          

acc = accuracy_score(ytest, y_pred)    # accuracy

# probability SVC returns
prob = svclassifier.predict_proba(Xtest)

end = time.time()

t = end - start


#######################  use probabilities ##########






