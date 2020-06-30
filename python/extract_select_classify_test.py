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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


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
################# extract feature ######################################
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

#var_mean = np.mean(variance,1).reshape(n,1)
K_mean = np.mean(K,1).reshape(n,1)  # choose kurtosis mean of 4 channels feature data
S_mean = np.mean(S,1).reshape(n,1)
act_mean = np.mean(activity,1).reshape(n,1)
mor_mean = np.mean(morbidity,1).reshape(n,1)
comp_mean = np.mean(complexity,1).reshape(n,1)
zc_mean = np.mean(zc,1).reshape(n,1)
dfa_mean = np.mean(d_f_a,1).reshape(n,1)
feature_matrix = np.hstack((K_mean,S_mean,act_mean,mor_mean,comp_mean,zc_mean,dfa_mean))


####################### Frequency domain  ################################

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

delta_mean = np.mean(delta_power,1).reshape(n,1)
theta_mean = np.mean(theta_power,1).reshape(n,1)
alpha_mean = np.mean(alpha_power,1).reshape(n,1)
beta_mean = np.mean(beta_power,1).reshape(n,1)
gamma_mean = np.mean(gamma_power,1).reshape(n,1)


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
# first order AR parameter (one parameter per epoch per channel)
AR_parameter = np.zeros((n,4))
for ii in range(n):
    for k in range(4):
        AR_parameter[ii,k] = AR_para[ii][k][0]
AR_mean = np.mean(AR_parameter,1).reshape(n,1)


########  Spectral edge frequency (SEF) for each band in each epoch
# 变量 Freq 范围，power百分比
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
 
sef50_delta_beta_mean = np.mean(sef50_delta_beta,1).reshape(n,1)
sef50_delta_alpha_mean = np.mean(sef50_delta_alpha,1).reshape(n,1)
sef50_alpha_mean = np.mean(sef50_alpha,1).reshape(n,1)
sef50_beta_mean = np.mean(sef50_beta,1).reshape(n,1)

feature_matrix_freq = np.hstack((delta_mean,theta_mean,alpha_mean,beta_mean,gamma_mean,AR_mean,sef50_delta_beta_mean,sef50_delta_alpha_mean,sef50_alpha_mean,sef50_beta_mean))

'''
######################### f-test ########################################
'''
# TIME
label = data['hypnogram']
y = label[1:90,0]   # label  remove the first epoch

f_time,p = f_classif(feature_matrix,y)
'''
selector = SelectKBest(f_classif, k=4)
selector.fit(feature_matrix, y)
feature_new = selector.transform(feature_matrix)
feature_label = selector.get_support(indices=True)
'''

# FREQUENCY
f_freq,p = f_classif(feature_matrix_freq,y)



'''
###################### SVM Classfication ##################################
'''

#### try time domain features first
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size = 0.20,random_state=0)
svclassifier = SVC(gamma=20,kernel='rbf',decision_function_shape='ovo')
#svclassifier = SVC(gamma=1,kernel='poly', degree=8)

svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#### SVM with univariate feature selection
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


clf = Pipeline([('anova', SelectPercentile(f_classif)),
                ('scaler', StandardScaler()),
                ('svc', SVC(gamma="auto"))])

score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, feature_matrix, y)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())


plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title(
    'Performance of the SVM-Anova varying the percentile of features selected')
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel('Percentile')
plt.ylabel('Accuracy Score')




