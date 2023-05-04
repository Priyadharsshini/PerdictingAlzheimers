
#  Author: Priyadharsshini Sakrapani

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix


import warnings
warnings.filterwarnings('ignore')

def Q1_results():
    global linear_performance
	# Load the train and test dataset
    train_sNC = pd.read_csv('train.fdg_pet.sNC.csv')
    train_sDAT = pd.read_csv('train.fdg_pet.sDAT.csv')

    test_sNC = pd.read_csv('test.fdg_pet.sNC.csv')
    test_sDAT = pd.read_csv('test.fdg_pet.sDAT.csv')

    # Create a column called 'group' which has sNC value as 0 and SDAT values as 1
    train_df = pd.concat([train_sNC, train_sDAT], ignore_index=True)
    train_df['group'] = [0] * len(train_sNC) + [1] * len(train_sDAT)

    # Handling null values as they are causing error and shuffle to make sure the prediction is not biased
    train_df = train_df.fillna(train_df.mean())

    # Split the data into feature and target
    X_train = train_df.iloc[:, :-2].values
    y_train = train_df.iloc[:, -1].values

    # Split the train dataset into training and validation sets
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold =  X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold =  y_train[train_index], y_train[val_index]

    # Define values for C to explore the performance
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 1000]}

    # Train and evaluate a linear SVM model for each value of C
    clf = GridSearchCV(LinearSVC(), param_grid, cv=skf, scoring=['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'], refit='accuracy')
    clf.fit(X_train, y_train)

    # Select the value of C that gives the best performance
    best_C = clf.best_params_['C']
    print("The best C is")
    print(best_C)

    # Train a final linear SVM model using the entire training dataset from the selected value of C
    final_clf = LinearSVC(C=best_C)
    final_clf.fit(X_train, y_train)

    # Prepare test data
    test_df = pd.concat([test_sNC, test_sDAT], ignore_index=True)
    test_df['group'] = [0] * len(test_sNC) + [1] * len(test_sDAT)
    test_df = test_df.fillna(test_df.mean())
    X_test = test_df.iloc[:, :-2].values
    y_test = test_df.iloc[:, -1].values

    # Evaluate the performance of the final model on the test dataset
    y_pred = final_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Calculate sensitivity (recall) and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Print the Err that we received from the previous step on the test data
    linear_performance = {'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'balanced_accuracy': bal_acc, 
                    'sensitivity': sensitivity,
                    'specificity':specificity}
    print("Performance of the final model on the test dataset")
    for metric, score in linear_performance.items():
        print(f'{metric}: {score:.4f}')

    print("\n")
    # Print the performance of the models explored during “C” hyperparameter tuning phase as function of “C”
    results = pd.DataFrame(clf.cv_results_)
    print(results[['param_C', 'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1', 'mean_test_balanced_accuracy']])

    # Plot the performance of the models explored during “C” hyperparameter tuning phase as function of “C”
    plt.figure(figsize=(10, 6))
    plt.plot(param_grid['C'], clf.cv_results_['mean_test_accuracy'], label='Accuracy')
    plt.plot(param_grid['C'], clf.cv_results_['mean_test_precision'], label='Precision')
    plt.plot(param_grid['C'], clf.cv_results_['mean_test_recall'], label='Recall')
    plt.plot(param_grid['C'], clf.cv_results_['mean_test_f1'], label='F1')
    plt.plot(param_grid['C'], clf.cv_results_['mean_test_balanced_accuracy'], label='Balanced Accuracy')
    plt.xlabel('C')
    plt.ylabel('Score')
    plt.title('Performance of linear SVM with different values of C')
    plt.legend()
    plt.show()

def Q2_results():
    global polynomial_kernel_performance
    # Load the train and test dataset
    train_sNC = pd.read_csv('train.fdg_pet.sNC.csv')
    train_sDAT = pd.read_csv('train.fdg_pet.sDAT.csv')

    test_sNC = pd.read_csv('test.fdg_pet.sNC.csv')
    test_sDAT = pd.read_csv('test.fdg_pet.sDAT.csv')

    # Create a column called 'group' which has sNC value as 0 and SDAT values as 1
    train_df = pd.concat([train_sNC, train_sDAT], ignore_index=True)
    train_df['group'] = [0] * len(train_sNC) + [1] * len(train_sDAT)

    # Handling null values as they are causing error and shuffle to make sure the prediction is not biased
    train_df = train_df.fillna(train_df.mean())

    # Split the data into feature and target
    X_train = train_df.iloc[:, :-2].values
    y_train = train_df.iloc[:, -1].values

    # Split the dataset into training and validation sets
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Define values for C and degree of the polynomial kernel d
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 1000], 'degree': [2, 3, 4, 5]}

    # Train and evaluate a polynomial kernel SVM model for each combination of C and d
    clf = GridSearchCV(SVC(kernel='poly'), param_grid, cv=skf, scoring=['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'], refit='accuracy')
    clf.fit(X_train, y_train)

    # Select the values of C and d that give the best performance on the validation set
    best_C = clf.best_params_['C']
    best_d = clf.best_params_['degree']
    print("The best C is")
    print(best_C)
    print("\n")
    print("The best D is")
    print(best_d)

    # Train a final polynomial kernel SVM model using the entire training dataset and the selected values of C and d
    final_clf = SVC(kernel='poly', C=best_C, degree=best_d)
    final_clf.fit(X_train, y_train)

    # Prepate the test dataset
    test_df = pd.concat([test_sNC, test_sDAT], ignore_index=True)
    test_df['group'] = [0] * len(test_sNC) + [1] * len(test_sDAT)
    test_df = test_df.fillna(test_df.mean())
    X_test = test_df.iloc[:, :-2].values
    y_test = test_df.iloc[:, -1].values

    # Evaluate the performance of the final model on the test dataset
    y_pred = final_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Calculate sensitivity (recall) and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Print the Err that we received from the previous step on the test data
    polynomial_kernel_performance = {'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'balanced_accuracy': bal_acc, 
                    'sensitivity': sensitivity,
                    'specificity':specificity}


    print("Performance of the final model on the test dataset")
    for metric, score in polynomial_kernel_performance.items():
        print(f'{metric}: {score:.4f}')

    print("\n")

    # Compare the performance of the polynomial kernel SVM model with that of the final linear SVM model
    print('Final Linear SVM Performance:')
    for metric, score in linear_performance.items():
        print(f'{metric}: {score:.4f}')

    print('\nFinal Polynomial SVM Performance:')
    for metric, score in polynomial_kernel_performance.items():
        print(f'{metric}: {score:.4f}')

    # plot the comparison of performance metrics
    x = ['Linear SVM', 'Polynomial SVM']
    accuracy = [linear_performance['accuracy'], polynomial_kernel_performance['accuracy']]
    precision = [linear_performance['precision'], polynomial_kernel_performance['precision']]
    recall = [linear_performance['recall'], polynomial_kernel_performance['recall']]
    # f1_score = [linear_performance['f1_score'], polynomial_kernel_performance['f1_score']]
    balanced_accuracy = [linear_performance['balanced_accuracy'], polynomial_kernel_performance['balanced_accuracy']]
    sensitivity = [linear_performance['sensitivity'], polynomial_kernel_performance['sensitivity']]
    specificity = [linear_performance['specificity'], polynomial_kernel_performance['specificity']]

    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    ax[0, 0].bar(x, accuracy)
    ax[0, 0].set_ylabel('Accuracy')
    ax[0, 0].set_title('Comparison of Linear and Polynomial SVMs')

    ax[0, 1].bar(x, precision)
    ax[0, 1].set_ylabel('Precision')
    ax[0, 1].set_title('Comparison of Linear and Polynomial SVMs')

    ax[1, 0].bar(x, recall)
    ax[1, 0].set_ylabel('Recall')
    ax[1, 0].set_title('Comparison of Linear and Polynomial SVMs')

    ax[1, 1].bar(x, balanced_accuracy)
    ax[1, 1].set_ylabel('Balanced Accuracy')
    ax[1, 1].set_title('Comparison of Linear and Polynomial SVMs')

    ax[2, 0].bar(x, sensitivity)
    ax[2, 0].set_ylabel('Sensitivity')
    ax[2, 0].set_title('Comparison of Linear and Polynomial SVMs')

    ax[2, 1].bar(x, specificity)
    ax[2, 1].set_ylabel('Specificity')
    ax[2, 1].set_title('Comparison of Linear and Polynomial SVMs')


    plt.tight_layout()
    plt.show()

def Q3_results():
    global RBF_kernel_performance
    # Load the train and test dataset
    train_sNC = pd.read_csv('train.fdg_pet.sNC.csv')
    train_sDAT = pd.read_csv('train.fdg_pet.sDAT.csv')

    test_sNC = pd.read_csv('test.fdg_pet.sNC.csv')
    test_sDAT = pd.read_csv('test.fdg_pet.sDAT.csv')

    # Create a column called 'group' which has sNC value as 0 and SDAT values as 1
    train_df = pd.concat([train_sNC, train_sDAT], ignore_index=True)
    train_df['group'] = [0] * len(train_sNC) + [1] * len(train_sDAT)

    # Handling null values as they are causing error and shuffle to make sure the prediction is not biased
    train_df = train_df.fillna(train_df.mean())

    # Split the data into feature and target
    X_train = train_df.iloc[:, :-2].values
    y_train = train_df.iloc[:, -1].values

    # Split the dataset into training and validation sets
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Define a range of values for the regularization parameter C and gamma parameter of the RBF kernel
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 300, 500, 1000]}

    # Train and evaluate an RBF kernel SVM model for each combination of C and gamma in the range
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=skf, scoring=['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'], refit='accuracy')
    clf.fit(X_train, y_train)

    # Select the values of C and gamma that give the best performance on the validation set
    best_C = clf.best_params_['C']
    best_gamma = clf.best_params_['gamma']
    print("The best C is")
    print(best_C)
    print("\n")
    print("The best gamma is")
    print(best_gamma)

    # Train a final RBF kernel SVM model using the entire training dataset and the selected values of C and gamma
    final_clf = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    final_clf.fit(X_train, y_train)

    # Prepare the test data
    test_df = pd.concat([test_sNC, test_sDAT], ignore_index=True)
    test_df['group'] = [0] * len(test_sNC) + [1] * len(test_sDAT)
    test_df = test_df.fillna(test_df.mean())
    X_test = test_df.iloc[:, :-2].values
    y_test = test_df.iloc[:, -1].values

    # Evaluate the performance of the final model on the test dataset
    y_pred = final_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Calculate sensitivity (recall) and specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    # Print the Err that we received from the previous step on the test data
    RBF_kernel_performance = {'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'balanced_accuracy': bal_acc, 
                    'sensitivity': sensitivity,
                    'specificity':specificity}

    print("Performance of the final model on the test dataset")
    for metric, score in RBF_kernel_performance.items():
        print(f'{metric}: {score:.4f}')

    print("\n")

    # Step 9: Compare the performance of the polynomial kernel SVM model with that of the final linear SVM model
    print('Final Linear SVM Performance:')
    for metric, score in linear_performance.items():
        print(f'{metric}: {score:.4f}')

    print('\nFinal Polynomial SVM Performance:')
    for metric, score in polynomial_kernel_performance.items():
        print(f'{metric}: {score:.4f}')

    print('\nFinal RBF Kernal SVM Performance:')
    for metric, score in RBF_kernel_performance.items():
        print(f'{metric}: {score:.4f}')

    # plot the comparison of performance metrics
    x = ['Linear SVM', 'Polynomial SVM', 'RBF Kernal SVM']
    accuracy = [linear_performance['accuracy'], polynomial_kernel_performance['accuracy'], RBF_kernel_performance['accuracy']]
    precision = [linear_performance['precision'], polynomial_kernel_performance['precision'], RBF_kernel_performance['precision']]
    recall = [linear_performance['recall'], polynomial_kernel_performance['recall'], RBF_kernel_performance['recall']]
    balanced_accuracy = [linear_performance['balanced_accuracy'], polynomial_kernel_performance['balanced_accuracy'], RBF_kernel_performance['balanced_accuracy']]
    sensitivity = [linear_performance['sensitivity'], polynomial_kernel_performance['sensitivity'], RBF_kernel_performance['sensitivity']]
    specificity = [linear_performance['specificity'], polynomial_kernel_performance['specificity'], RBF_kernel_performance['specificity']]


    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    ax[0, 0].bar(x, accuracy)
    ax[0, 0].set_ylabel('Accuracy')
    ax[0, 0].set_title('Comparison of Linear, Polynomial, RBF kernel SVMs')

    ax[0, 1].bar(x, precision)
    ax[0, 1].set_ylabel('Precision')
    ax[0, 1].set_title('Comparison of Linear, Polynomial, RBF kernel SVMs')

    ax[1, 0].bar(x, recall)
    ax[1, 0].set_ylabel('Recall')
    ax[1, 0].set_title('Comparison of Linear, Polynomial, RBF kernel SVMs')

    ax[1, 1].bar(x, balanced_accuracy)
    ax[1, 1].set_ylabel('Balanced Accuracy')
    ax[1, 1].set_title('Comparison of Linear, Polynomial, RBF kernel SVMs')

    ax[2, 0].bar(x, sensitivity)
    ax[2, 0].set_ylabel('Sensitivity')
    ax[2, 0].set_title('Comparison of Linear, Polynomial, RBF kernel SVMs')

    ax[2, 1].bar(x, specificity)
    ax[2, 1].set_ylabel('Specificity')
    ax[2, 1].set_title('Comparison of Linear, Polynomial, RBF kernel SVMs')

    plt.tight_layout()
    plt.show()

def diagnoseDAT(Xtest, data_dir):
    # Load the training data
    sNC_train = pd.read_csv(data_dir + "/train.fdg_pet.sNC.csv")
    sDAT_train = pd.read_csv(data_dir + "/train.fdg_pet.sDAT.csv")

    # Separate the features and labels
    train_df = pd.concat([sNC_train, sDAT_train], ignore_index=True)
    train_df['group'] = [0] * len(sNC_train) + [1] * len(sDAT_train)

    # Handling null values as they are causing error and shuffle to make sure the prediction is not biased
    train_df = train_df.fillna(train_df.mean())
    

    # Split the data into feature and target
    X_train = train_df.iloc[:, :-2].values
    y_train = train_df.iloc[:, -1].values

    # Data preprocessing: normalize the features using standardization
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)

    # Feature selection: select the glucose metabolism features derived from the 14 cortical brain regions
    X_train = X_train[:, :28]

    # Fit the SVM model
    svm = SVC(kernel='rbf', C=10, gamma=10)
    svm.fit(X_train, y_train)
    Xtest = Xtest.fillna(Xtest.mean())

    # Preprocess the test data
    Xtest = Xtest.iloc[:, :-2].values
    

    Xtest = Xtest[:, :28]

    # Xtest = scaler.transform(Xtest)
   

    # Make predictions on the test set
    ytest = svm.predict(Xtest)

    return ytest

if __name__ == "__main__":  
    Q1_results()
    Q2_results()
    Q3_results()
    # diagnoseDAT(test_df,"file:///Users/priyadharsshinis/" )




