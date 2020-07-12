import numpy as np
from scipy.io import loadmat,savemat
import time
#from scipy import signal
import matplotlib.pyplot as plt
#from scipy import integrate
#from sklearn.feature_selection import  f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


feature_data = loadmat('feature_data.mat')
feature_overlap = loadmat('feature_overlap_data.mat')
label = feature_data['label'].ravel()
feature_matrix_selected = feature_data['feature_selected']
transition_matrix = feature_data['transition_matrix']
feature_overlap_selected = feature_overlap['feature_overlap_selected']


'''
########################################## SVM ###############################
'''

number = range(len(label))
number = np.asarray(number)
Xtrain, Xtest, ytrain_idx, ytest_idx = train_test_split(feature_matrix_selected, number, test_size = 0.2)

ytrain = label[ytrain_idx]
ytest = label[ytest_idx]

#svclassifier = SVC(gamma=0.000005,C = 200, kernel='rbf',decision_function_shape='ovr',probability=True)
svclassifier = SVC(gamma=0.000005,C = 200, kernel='rbf',decision_function_shape='ovr')
svclassifier.fit(Xtrain, ytrain.ravel())
y_pred = svclassifier.predict(Xtest)

print(confusion_matrix(ytest, y_pred))
print(classification_report(ytest, y_pred))          

acc1 = accuracy_score(ytest, y_pred)    # accuracy

# probability SVC returns by decision function
p = np.array(svclassifier.decision_function(Xtest)) # decision is a voting function
prob = np.exp(p)/(np.sum(np.exp(p),axis=1).reshape((len(ytest_idx),1)))


######################## classify by conditoinal probabilities

test_idx_previous = ytest_idx - 1    # previous states idx of test data 
label_previous = label[test_idx_previous]

con_prob = np.zeros((len(ytest),4))
for test_num in range(int(len(ytest))):
    previous_state = label_previous[test_num]  # 0,1,2,3
    # compute conditional prob for 4 possible current states    
    for ii in range(4):
        con_prob[test_num,ii] = transition_matrix[previous_state,ii] * prob[test_num,ii]

y_pred_con = np.argmax(con_prob,axis = 1)
acc2 = accuracy_score(ytest, y_pred_con)

print(confusion_matrix(ytest, y_pred_con))
print(classification_report(ytest, y_pred_con))


'''
################### overlapping epochs SVM ###############################
'''
train_data = feature_matrix_selected
train_label = label
test_data = feature_overlap_selected

svclassifier.fit(train_data, train_label)
y_overlap_pred = svclassifier.predict(test_data)  # predict overlap epochs


'''
###################### plot ########################
'''

x_overlap = np.arange(1,len(y_overlap_pred)+1)
plt.plot(x_overlap,y_overlap_pred)
plt.legend()
plt.grid()
plt.show()


x = np.arange(1,len(train_label)+1)
plt.plot(x,train_label)
plt.legend()
plt.grid()
plt.show()


