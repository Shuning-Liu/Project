import numpy as np
from sklearn import tree
import numpy as np
from scipy.io import loadmat,savemat
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# H 测试样本分类结果
# TrainS 原训练样本 np数组
# TrainA 辅助训练样本
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数
def tradaboost(trans_S, trans_A, label_S, label_A, test, N):
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_label = np.concatenate((label_A, label_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    test_data = np.concatenate((trans_data, test), axis=0)

    # 初始化权重
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    #bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))
    bata = 1 / (1 + np.sqrt(2 * np.log(row_A) / N))

    # 存储每次迭代的标签和bata值？
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    predict = np.zeros([row_T])

    #print ('params initial finished.')
    trans_data = np.asarray(trans_data, order='C')
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        P = calculate_P(weights, trans_label)

        result_label[:, i] = train_classify(trans_data, trans_label,
                                            test_data, P)
        #print ('result,', result_label[:, i], row_A, row_S, i, result_label.shape)

        error_rate = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
                                          weights[row_A:row_A + row_S, :])
        print ('Error rate:', error_rate)
        if error_rate > 0.5:
            error_rate = 0.5
        if error_rate == 0:
            N = i
            break  # 防止过拟合
            # error_rate = 0.001

        bata_T[0, i] = error_rate / (1 - error_rate)

        # 调整源域样本权重
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                                               (-np.abs(result_label[row_A + j, i] - label_S[j])))

        # 调整辅域样本权重
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))
    # print bata_T
    for i in range(row_T):
        # 跳过训练数据的标签
        left = np.sum(
            result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
            # print left, right, predict[i]

    return predict,result_label


def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classify(trans_data, trans_label, test_data, P):
    
    #clf = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
    #clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    clf = SVC(gamma=0.0005,C = 200, kernel='rbf',decision_function_shape='ovr')
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    
    return clf.predict(test_data)


def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)

    #print (weight[:, 0] / total)
    #print (np.abs(label_R - label_H))
    return np.sum(weight[:, 0] / total * np.abs(label_R - label_H))




feature_data = loadmat('feature_data.mat')
label = feature_data['label'].ravel()
feature_matrix_selected = feature_data['feature_selected']
transition_matrix = feature_data['transition_matrix']

#Xtrain = feature_matrix_selected[0:89+89+89,:]
#ytrain = label[0:89+89+89]
Xtest = feature_matrix_selected[89+89+89:330,:]
ytest = label[89+89+89:]

#XtrainA, XtrainB, ytrainA, ytrainB = train_test_split(Xtrain, ytrain, test_size = 0.4)
XtrainB = feature_matrix_selected[89+89+63:89+89+89,:]
ytrainB = label[89+89+63:89+89+89]
XtrainA = feature_matrix_selected[0:89+89+63,:]
ytrainA = label[0:89+89+63]

pp,result_label = tradaboost(XtrainA, XtrainB, ytrainA, ytrainB, Xtest, 10)

#fpr, tpr, thresholds = metrics.roc_curve(y_true=ytest, y_score=pp, pos_label=1)











