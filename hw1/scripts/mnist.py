import sys

if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from scipy import io
from save_csv import results_to_csv

# Data MNIST
def data_partition_mnist(x,y):
    totol_size=x.shape[0]
    indices=np.arange(totol_size)
    np.random.shuffle(indices)

    val_num=10000
    val_indices=indices[:val_num]
    train_indices=indices[val_num:]

    # validate data&labels
    x_val=x[val_indices]
    y_val=y[val_indices]

    # train data&labels
    x_train=x[train_indices]
    y_train=y[train_indices]
    
    return x_train,y_train,x_val,y_val

#  y: true labels,y_pred: predicted labels
def evaluate_metric(y,y_pred):
    accuracy_score=0
    counter=0
    n=y.shape[0]
    if y.shape[0] != y_pred.shape[0]:
        raise ValueError("shape is not matched")
    return  np.mean(y==y_pred)

# For each dataset, plot the classification accuracy
# if not do that,hyperparameters will not work
def do_standardize(test_data, train_data):
    test_data = test_data.reshape(len(test_data), -1)
    train_data = train_data.reshape(len(train_data), -1)
    index = len(test_data)
    print(index)
    data = np.append(test_data, train_data, axis=0)
    data = (data - np.min(data))/(np.max(data) - np.min(data))
    return data[0: index, :], data[index::, :]



if __name__=="__main__":
    data_mnist=np.load(f"../data/{"mnist"}-data.npz")
    test_data = data_mnist["test_data"]
    train_data = data_mnist['training_data']
    test_data, train_data = do_standardize(test_data, train_data)
    x_train_mnist,y_train_mnist,x_val_mnist,y_val_mnist=data_partition_mnist(train_data,data_mnist["training_labels"])
    x_train_mnist = x_train_mnist.reshape(x_train_mnist.shape[0], -1)
    y_train_mnist = y_train_mnist
    x_val_mnist=x_val_mnist.reshape(x_val_mnist.shape[0],-1)
    y_val_mnist=y_val_mnist.ravel()
    
    train_acc_list = []
    val_acc_list = []    
    
    # model=svm.SVC(kernel="linear",C=7.5,gamma=0.03) Train acc: 0.93066
    model = svm.SVC(C=7.5, kernel="poly", degree=3) # 
    model.fit(x_train_mnist,y_train_mnist)
    train_acc=evaluate_metric(y_train_mnist,model.predict(x_train_mnist))
    val_acc=evaluate_metric(y_val_mnist,model.predict(x_val_mnist))
    
    print("\tTrain acc: {}".format(train_acc))
    print("\tVal acc: {}".format(val_acc))
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    results_to_csv(model.predict(test_data),"mnist") 


