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
label = feature_data['label'].ravel()
feature_matrix_selected = feature_data['feature_selected']
transition_matrix = feature_data['transition_matrix']



'''
########################################## SVM ###############################
'''
Xtrain = feature_matrix_selected[0:89+89+63,:]
ytrain = label[0:89+89+63]
Xtest = feature_matrix_selected[89+89+63:330,:]
ytest = label[89+89+63:]

svclassifier = SVC(gamma=0.000001,C = 200, kernel='rbf',decision_function_shape='ovr')
svclassifier.fit(Xtrain, ytrain.ravel())
y_pred = svclassifier.predict(Xtest)
print(confusion_matrix(ytest, y_pred))
print(classification_report(ytest, y_pred))          

acc1 = accuracy_score(ytest, y_pred)    # accuracy

# probability SVC returns by decision function
p = np.array(svclassifier.decision_function(Xtest)) # decision is a voting function
prob = np.exp(p)/(np.sum(np.exp(p),axis=1).reshape((len(ytest),1)))


label_previous = label[89+89+62:329]
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







