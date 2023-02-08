import sklearn
import numpy
import random
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
import statistics
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import time

import warnings
warnings.filterwarnings('ignore')

def evaluate_score(output,test_label):
    accuracy=metrics.accuracy_score(test_label, output)
    error = 1 - accuracy
    return error, accuracy
    #error is equal to accuracy - 1 which is also same as 1-the fraction of misclassified cases
    #creating the mean of each error value and making an array
    #https://stats.stackexchange.com/questions/133458/is-accuracy-1-test-error-rate

def train(input_data,input_label,kernel_type_lin,kernel_type_rbf):
    all_error = []
    errork1=[]
    errork5=[]
    errork10=[]
    error_svm_linear=[]
    error_svm_rbf=[]
    error_random_forest=[]
    accuracyk1=[]
    accuracyk5=[]
    accuracyk10=[]
    accuracy_svm_linear=[]
    accuracy_svm_rbf=[]
    accuracy_random_forest=[]
    StratifiedK = StratifiedShuffleSplit(n_splits=5,train_size=0.8,test_size=0.2, random_state=0)    
    StratifiedK.get_n_splits(input_data, input_label)

    for data_train, data_test in StratifiedK.split(input_data, input_label):
         
        training_set, testing_set = input_data[data_train], input_data[data_test]
        train_label, test_label = input_label[data_train],input_label[data_test]

        knn1 = KNeighborsClassifier(n_neighbors=1)
        knn5 = KNeighborsClassifier(n_neighbors=5)
        knn10 = KNeighborsClassifier(n_neighbors=10)
        ran_clf = RandomForestClassifier(n_estimators=100)
        svm_lin_clf = svm.SVC(kernel=kernel_type_lin, gamma='scale')
        svm_rbf_clf = svm.SVC(kernel=kernel_type_rbf, gamma='scale')

        start_time = time.time()
        svm_lin_clf.fit(training_set, train_label)
        svm_lin_model = 'svm_lin_model.sav'
        pickle.dump(svm_lin_clf, open(svm_lin_model, 'wb'))
        loaded_model = pickle.load(open('svm_lin_model.sav', 'rb'))
        output_svm_lin = svm_lin_clf.predict(testing_set)

        svm_rbf_clf.fit(training_set, train_label)
        svm_rbf_model = 'svm_rbf_model.sav'
        pickle.dump(svm_rbf_clf, open(svm_rbf_model, 'wb'))
        output_svm_rbf = svm_rbf_clf.predict(testing_set)

        ran_clf.fit(training_set, train_label)
        ran_clf_model = 'ran_clf_model.sav'
        pickle.dump(ran_clf, open(ran_clf_model, 'wb'))
        ran_output = ran_clf.predict(testing_set)

        knn1.fit(training_set, train_label)
        knn1_model = 'knn1_model.sav'
        pickle.dump(knn1, open(knn1_model, 'wb'))
        outputk1 = knn1.predict(testing_set)

        knn5.fit(training_set, train_label)
        knn5_model = 'knn5_model.sav'
        pickle.dump(knn5_model, open(knn5_model, 'wb'))
        outputk5 = knn5.predict(testing_set)

        knn10.fit(training_set, train_label)
        knn10_model = 'knn10_model.sav'
        pickle.dump(knn10_model, open(knn10_model, 'wb'))
        outputk10 = knn10.predict(testing_set)

        end_time = time.time()
        knn_train_time = end_time - start_time
        print(f"Time before and during training for kNN model: {knn_train_time:.2f} seconds")

        error_svm_lin, accuracy_svm_lin = evaluate_score(output_svm_lin,test_label)
        error_svm_linear.append(error_svm_lin)
        accuracy_svm_linear.append(accuracy_svm_lin)


        error_svm_rbf1, accuracy_svm_rbf1 = evaluate_score(output_svm_rbf,test_label)
        error_svm_rbf.append(error_svm_rbf1)
        accuracy_svm_rbf.append(accuracy_svm_rbf1)


        error_random_forest1, accuracy_random_forest1 = evaluate_score(ran_output,test_label)
        error_random_forest.append(error_random_forest1)
        accuracy_random_forest.append(accuracy_random_forest1)


        err_k1, acc_k1 = evaluate_score(outputk1,test_label)
        errork1.append(err_k1)
        accuracyk1.append(acc_k1)


        err_k5, acc_k5 = evaluate_score(outputk5,test_label)
        errork5.append(err_k5)
        accuracyk5.append(acc_k5)


        err_k10, acc_k10 = evaluate_score(outputk10,test_label)
        errork10.append(err_k10)
        accuracyk10.append(acc_k10)

    all_error.append(statistics.mean(error_svm_linear))
    all_error.append(statistics.mean(accuracy_svm_linear))
    all_error.append(statistics.mean(error_svm_rbf))
    all_error.append(statistics.mean(accuracy_svm_rbf))
    all_error.append(statistics.mean(error_random_forest))
    all_error.append(statistics.mean(accuracy_random_forest))
    all_error.append(statistics.mean(errork1))
    all_error.append(statistics.mean(accuracyk1))
    all_error.append(statistics.mean(errork5))
    all_error.append(statistics.mean(accuracyk5))
    all_error.append(statistics.mean(errork10))
    all_error.append(statistics.mean(accuracyk10))
    print(all_error)
    # creating the array with the error of each classifier.
    #https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb
    #https://www.geeksforgeeks.org/python-statistics-mean-function/
    #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    #https://scikit-learn.org/stable/modules/svm.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    return all_error



if __name__ == '__main__':
    data = pd.read_csv('dataset.csv')
    # data = firstdata.sample(8000,random_state=2)

    # The percentage of the number of rows to select (between 0 and 1)
    percentage = 0.4

    # Number of rows to select 
    num_rows = int(len(data) * percentage)

    # Generate a list of random indices
    random_indices = random.sample(range(0, len(data)), num_rows)

    # Select the rows with the corresponding indices
    random_rows = data.iloc[random_indices]

    # Create a new DataFrame with the selected rows
    data_set = pd.DataFrame(random_rows)

    print(data_set.shape)
    final_df = data_set.drop(columns=['id','qid1','qid2','question1','question2'])
    # print(final_df.shape)

    input_label = numpy.array(final_df.iloc[:, 1])
    # input_label = input_label.to_frame()
    input_data = numpy.array(final_df.drop(columns=['is_duplicate', 'Unnamed: 0']))
    all_error = train(input_data,input_label,'linear','rbf')



