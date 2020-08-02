import numpy as np
from scipy.io import loadmat,savemat

import time
from scipy import signal
import matplotlib.pyplot as plt
from scipy import integrate
from sklearn.feature_selection import  f_classif

def two_state_fscore(a,b,feature_matrix,label):
    idx_ab = np.where((label == a) | (label == b))[0]
    label_ab = label[idx_ab]
    feature_ab = feature_matrix[idx_ab,:]
    f,p = f_classif(feature_ab,label_ab)
    f_ab = np.zeros(int(len(f)/4))
    for ii in range(int(len(f)/4)):
        f_ab[ii] = max(f[ii*4:(ii+1)*4])
    return f_ab

feature_data = loadmat('feature_data.mat')
label = feature_data['label'].ravel()
feature_total = feature_data['feature_total']

feature_1 = feature_total[0:89,:]
feature_2 = feature_total[89:89*2,:]
feature_3 = feature_total[89*2:89*2+63,:]
feature_4 = feature_total[89*2+63:,:]
feature = [feature_1,feature_2,feature_3,feature_4]
label_1 = label[0:89]
label_2 = label[89:89*2]
label_3 = label[89*2:89*2+63]
label_4 = label[89*2+63:]
label = [label_1,label_2,label_3,label_4]

f_two = []
for ii in range(4):
    f_01 = two_state_fscore(0,1,feature[ii],label[ii])   # state W vs N1
    f_02 = two_state_fscore(0,2,feature[ii],label[ii])   # state W vs N2
    f_03 = two_state_fscore(0,3,feature[ii],label[ii])   # state W vs N3
    f_12 = two_state_fscore(1,2,feature[ii],label[ii])   # state N1 vs N2
    f_13 = two_state_fscore(1,3,feature[ii],label[ii])   # state N1 vs N3
    f_23 = two_state_fscore(2,3,feature[ii],label[ii])   # state N2 vs N3
    f_two.append(np.vstack((f_01,f_02,f_03,f_12,f_13,f_23)))


# feature variance of each session

var = np.zeros((4,15))
for ii in range(4):
    var_i = np.var(feature[ii],0)
    #var_i = np.zeros(15)
    for k in range(15):
        var[ii,k] = max(var_i[k*4:(k+1)*4])

variance = []
for ii in range(15):
    variance.append(var[:,ii])

del variance[5]  # remove zc
del variance[0]  # remove K

# box plot
#labels = ['K','S','act','mor','com','zc','dfa','delta','theta','alpha','beta','gamma','sef50','sef95','sefd']
labels = ['S','act','mor','com','dfa','delta','theta','alpha','beta','gamma','sef50','sef95','sefd']
plt.boxplot(variance, labels = labels, sym = "o")
plt.show()












