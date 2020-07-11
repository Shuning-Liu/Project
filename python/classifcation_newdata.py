import numpy as np
from scipy.io import loadmat,savemat
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score



feature_data = loadmat('feature_new_data.mat')
feature_selected = feature_data['feature_selected']
label = feature_data['label'].ravel()
transition_matrix = feature_data['transition_matrix']

'''
########################################## SVM ###############################
'''
number = range(len(label))
number = np.asarray(number)
Xtrain, Xtest, ytrain_idx, ytest_idx = train_test_split(feature_selected, number, test_size = 0.2)
ytrain = label[ytrain_idx]
ytest = label[ytest_idx]

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

con_prob = np.zeros((len(ytest),6))
for test_num in range(int(len(ytest))):
    previous_state = label_previous[test_num]  # 0,1,2,3
    # compute conditional prob for 4 possible current states    
    for ii in range(6):
        con_prob[test_num,ii] = transition_matrix[previous_state,ii] * prob[test_num,ii]

y_pred_con = np.argmax(con_prob,axis = 1)
acc2 = accuracy_score(ytest, y_pred_con)

print(confusion_matrix(ytest, y_pred_con))
print(classification_report(ytest, y_pred_con))



