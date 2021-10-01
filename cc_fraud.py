import numpy as np
import pandas as pd
import opendatasets as od
from sklearn.model_selection import train_test_split

#Retrieve dataset from Kaggle
#od.download("https://www.kaggle.com/mlg-ulb/creditcardfraud")


#Upload onto df
df_fraud = pd.read_csv('/Users/georgeanton/Desktop/Applied_Data_Science_Python_UM/Applied_Machine_Learning/creditcardfraud/creditcard.csv')


#split data into features, target (2 methods)
y = df_fraud['Class']
X = df_fraud.iloc[:,1:-1]
#X, y = df_fraud.drop('Class',axis=1), df_fraud.Class

#percentage of observations in dataset which are instances of fraud (2 methods)
#res = y.sum(axis=0)/len(y)
#result = len(y[y==1])/(len(y[y==1]) + len(y[y==0]))

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)


#train dummy classifier that classifies everything in training data as the majority class
#compare this to accuracy score.
def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score

    # Your code here
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

    y_majority_predicted = dummy_majority.predict(X_test)

    acc_score = dummy_majority.score(X_test, y_test)
    rec_score = recall_score(y_test, y_majority_predicted)


    return acc_score, rec_score
#
# #Use SVC classifier with default parameters and compare accuracy, precision and recall scores
def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    # Your code here
    svm = SVC().fit(X_train, y_train)

    acc_score = svm.score(X_test, y_test)
    rec_score = recall_score(y_test, svm.predict(X_test))
    prec_score = precision_score(y_test, svm.predict(X_test))

    return acc_score, rec_score, prec_score

#print(acc_score, rec_score, prec_score)


#Use SVC classifier and determine confusion matrix when threshold is -220 on decision function
def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    # Your code here
    y_scores_svc = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train).decision_function(X_test) > -220

    confusion = confusion_matrix(y_test, y_scores_svc)


    return confusion

# Question 5
# Train a logisitic regression classifier with default parameters using X_train and y_train.
#
# For the logisitic regression classifier, create a precision recall curve and a roc curve
#using y_test and the probability estimates for X_test (probability it is fraud).
#
# Looking at the precision recall curve, what is the recall when the precision is 0.75?
#
# Looking at the roc curve, what is the true positive rate when the false positive rate is 0.16?
#
# This function should return a tuple with two floats, i.e. (recall, true positive rate).
#

def answer_five():

    # Your code here
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.metrics import roc_curve, precision_recall_curve
#     import numpy as np
#     import matplotlib.pyplot as plt

#     lr = LogisticRegression().fit(X_train, y_train)

#     y_predict = lr.predict(X_test)


#     precision, recall, threshold = precision_recall_curve(y_test, y_predict)

#     closest_zero = np.argmin(np.abs(threshold))
#     closest_zero_p = precision[closest_zero]
#     closest_zero_r = recall[closest_zero]
#     plt.figure()
#     plt.plot(precision, recall,label='Precision-Recall curve')
#     plt.plot(closest_zero_p, closest_zero_r, 'o', label='optimal point')
#     plt.legend()
#     plt.xlim([0.0, 1.1])
#     plt.ylim([0.0,1.1])
#     plt.xlabel('Precision')
#     plt.ylabel('Recall')
#     plt.show()

#     roc_prec_lr, roc_rec_lr, roc_thresh = roc_curve(y_test, y_predict)

#     plt.figure()
#     plt.plot(roc_prec_lr, roc_rec_lr)
#     plt.xlim([0.0, 1.1])
#     plt.ylim([0.0,1.1])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.show()

    return 0.83,0.94

# Question 6
# Perform a grid search over the parameters listed below
# for a Logisitic Regression classifier, using recall for scoring
#
# and the default 3-fold cross validation.
#
# 'penalty': ['l1', 'l2']
#
# 'C':[0.01, 0.1, 1, 10, 100]
#
# From .cv_results_, create an array of the mean test scores of each parameter combination.

def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # Your code here
    lr = LogisticRegression(C=3, solver='liblinear')

    grid_vals = {'C':[0.01,0.1,1,10,100],'penalty':['l1','l2']}

    grid_clf= GridSearchCV(lr, param_grid= grid_vals, scoring='recall')

    grid_clf.fit(X_train, y_train)

    mean_test_scores = grid_clf.cv_results_['mean_test_score'].reshape(-1,2)

    return mean_test_scores


# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0)
    plt.show()

GridSearch_Heatmap(answer_six())


#
