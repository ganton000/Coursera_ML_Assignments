import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

#Reshape data to 2d array (single column 11 rows)
# X_test=X_test.reshape(-1,1)
# X_train = X_train.reshape(-1,1)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    plt.show()


# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
#part1_scatter()





#Part 1 Regression
#


# Question 1
# Write a function that fits a polynomial LinearRegression model on the training data X_train
# for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the
# polynomial features and then fit a linear regression model)
# For each model, find 100 predicted values over the interval x = 0 to 10
# (e.g. np.linspace(0,10,100)) and store this in a numpy array.
# The first row of this array should correspond to the output from the model trained on degree 1,
# the second row degree 3, the third row degree 6, and the fourth row degree 9.


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def answer_one(X_train,y_train):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    X_train = X_train.reshape(11,1)
    res = np.zeros((4,100))

    for i, degree in enumerate([1,3,6,9]):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train)
        linreg = LinearRegression().fit(X_poly, y_train)
        train_x_poly = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)))
        res[i,:] = train_x_poly



    return res

def plot_one(degree_predictions):
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()

plot_one(answer_one(X_train,y_train))


# Question 2
# Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 0 through 9.
# For each model compute the  R^2  (coefficient of determination) regression score on the training data
# as well as the the test data, and return both of these arrays in a tuple.
#
# This function should return one tuple of numpy arrays (r2_train, r2_test). Both arrays should have shape (10,)


def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    global X_train, X_test, y_train, y_test

    # Your code here
    r2_train = np.zeros(10)
    r2_test = np.zeros(10)
    for i in range(10):
        poly = PolynomialFeatures(degree=i)
        X_poly = poly.fit_transform(X_train.reshape(-1,1))
        X_test_poly = poly.transform(X_test.reshape(-1,1))
        linreg = LinearRegression().fit(X_poly, y_train)
        r2_train[i], r2_test[i] = linreg.score(X_poly, y_train), linreg.score(X_test_poly, y_test)

    res = (r2_train, r2_test)


    return res


assert answer_two()[0].shape == (10,)
assert answer_two()[1].shape == (10,)

print(answer_two())

# Question 3
# Based on the  R2R2  scores from question 2 (degree levels 0 through 9),
# what degree level corresponds to a model that is underfitting?
# What degree level corresponds to a model that is overfitting?
# What choice of degree level would provide a model
# with good generalization performance on this dataset?
#
# Hint: Try plotting the  R^2  scores from question 2
# to visualize the relationship between degree level and  R^2.
# Remember to comment out the import matplotlib line before submission.
#
# This function should return one tuple with the degree values in this order:
# (Underfitting, Overfitting, Good_Generalization).
# There might be multiple correct solutions, however,
# you only need to return one possible solution, for example, (1,2,3).



def plot_three(x,y):
    import matplotlib.pyplot as plt

    global X_train, X_test, y_train, y_test

    r2_train_scores, r2_test_scores = x,y
    plt.figure(figsize=(10,5))
#     plt.plot(X_train, y_train, r2_train_scores, 'o', label='Training Data', markersize=10)
#     plt.plot(X_test, y_test, r2_test_scores, 'o', label='Test Data', markersize=10)
#     for i in range(10):
    plt.scatter(np.linspace(0,10,10), r2_train_scores, label='R2_Training_Scores')
    plt.scatter(np.linspace(0,10,10), r2_test_scores, label='R2_Test_Scores')
    plt.legend()
    plt.ylim=(-1,2.5)
    plt.ylabel('R2_Scores')
    plt.xlabel('Polynomial Degrees')
    plt.show()

x,y = answer_two()
plot_three(x,y)

def answer_three(inp):

    # Your code here
    R2_train_scores, R2_test_scores = answer_two()
    diff = R2_train_scores - R2_test_scores

    #based on data the closest pair of R2 values are near 1 and thus both data sets fit the model well
    #thus I will use argmin() for this. Similar reasoning for overfitting shows we use argmax()
    good_gen = np.argmin(diff)
    overfitting = np.argmax(diff)

    #We notice from the graph that biggest difference in R2 values are underfitting,
    #thus can use argmin() to return the index. Similar reasoning for overfitting shows we use argmax()

    underfitting = 0

    return underfitting, overfitting, good_gen

print(answer_three(answer_two))

#
# Question 4
# Training models on high degree polynomial features can result in overly complex models that overfit,
# so we often use regularized versions of the model to constrain model complexity,
# as we saw with Ridge and Lasso linear regression.
#
# For this question, train two models: a non-regularized LinearRegression model (default parameters)
# and a regularized Lasso Regression model (with parameters alpha=0.01, max_iter=10000) both on polynomial features
# of degree 12. Return the  R2R2  score for both the LinearRegression and Lasso model's test sets.
#
# This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression

    global X_train, X_test, y_train, y_test

    poly = PolynomialFeatures(degree=12)
    X_poly = poly.fit_transform(X_train.reshape(-1,1))
    linreg = LinearRegression().fit(X_poly, y_train)
    lasso_reg = Lasso(alpha=0.01, max_iter=10000).fit(X_poly,y_train)

    X_test_poly = poly.transform(X_test.reshape(-1,1))
    LinearRegression_R2_test_score = linreg.score(X_test_poly,y_test)
    Lasso_R2_test_score = lasso_reg.score(X_test_poly,y_test)

    return LinearRegression_R2_test_score, Lasso_R2_test_score


print(answer_four())


# Part 2 - Classification
# Here's an application of machine learning that could save your life!
# For this section of the assignment we will be working with the
# UCI Mushroom Data Set stored in readonly/mushrooms.csv.
# The data will be used to train a model to predict whether or not a mushroom is poisonous.
# The following attributes are provided:
#
# Attribute Information:
#
# cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
# cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
# cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
# bruises?: bruises=t, no=f
# odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
# gill-attachment: attached=a, descending=d, free=f, notched=n
# gill-spacing: close=c, crowded=w, distant=d
# gill-size: broad=b, narrow=n
# gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
# stalk-shape: enlarging=e, tapering=t
# stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
# stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
# stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
# stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
# stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
# veil-type: partial=p, universal=u
# veil-color: brown=n, orange=o, white=w, yellow=y
# ring-number: none=n, one=o, two=t
# ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
# spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
# population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
# habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
#
#
# The data in the mushrooms dataset is currently encoded with strings.
# These values will need to be encoded to numeric to work with sklearn.
# We'll use pd.get_dummies to convert the categorical variables into indicator variables.


mush_df = pd.read_csv('/Users/georgeanton/Desktop/Applied_Data_Science_Python_UM/Applied_Machine_Learning/course3_downloads/mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1] #2nd column holds classification ('class' column)

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

# Question 5
# Using X_train2 and y_train2 from the preceeding cell,
# train a DecisionTreeClassifier with default parameters and random_state=0.
# What are the 5 most important features found by the decision tree?
#
# As a reminder, the feature names are available in the X_train2.columns property,
# and the order of the features in X_train2.columns matches the order of the feature importance
# values in the classifier's feature_importances_ property.
#
# This function should return a list of length 5 containing the feature names
# in descending order of importance.
#
# Note: remember that you also need to set random_state in the DecisionTreeClassifier.

def answer_five():
    from sklearn.tree import DecisionTreeClassifier
    from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

    # def plot_feature_importances(clf, feature_names):
    #     c_features = len(feature_names)
    #     plt.barh(range(c_features), clf.feature_importances_)
    #     plt.xlabel("Feature importance")
    #     plt.ylabel("Feature name")
    #     plt.yticks(np.arange(c_features), feature_names)

    global X_train2, X_test2, y_train2, y_test2

    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)

    features_list = []

    for i, importance_val in enumerate(clf.feature_importances_):
        features_list.append([importance_val, X_train2.columns[i]])


    top_five_features = sorted(features_list, reverse=1)[:5]
    top_five_names = [ top_five_features[i][1] for i in range(5)]


    # plot_feature_importances(clf, X_train2.columns)
    # plt.show()

    return top_five_names


# lst = answer_five()
# print(lst)

# Question 6
# For this question, we're going to use the validation_curve function in sklearn.model_selection
# to determine training and test scores for a Support Vector Classifier (SVC)
# with varying parameter values. Recall that the validation_curve function,
# in addition to taking an initialized unfitted classifier object,
# takes a dataset as input and does its own internal train-test splits to compute results.
#
# Because creating a validation curve requires fitting multiple models,
# for performance reasons this question will use just a subset of the original mushroom dataset:
# please use the variables X_subset and y_subset as input to the validation curve function
# (instead of X_mush and y_mush) to reduce computation time.
#
# The initialized unfitted classifier object we'll be using is
# a Support Vector Classifier with radial basis kernel.
# So your first step is to create an SVC object with default parameters
# (i.e. kernel='rbf', C=1) and random_state=0.
# Recall that the kernel width of the RBF kernel is controlled using the gamma parameter.
#
# With this classifier, and the dataset in X_subset, y_subset,
# explore the effect of gamma on classifier accuracy by using the validation_curve function
# to find the training and test scores for 6 values of gamma from 0.0001 to 10
# (i.e. np.logspace(-4,1,6)). Recall that you can specify what scoring metric you want
# validation_curve to use by setting the "scoring" parameter.
# In this case, we want to use "accuracy" as the scoring metric.
#
# For each level of gamma, validation_curve will fit 3 models on different subsets of the data,
# returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training
# and test sets.
#
# Find the mean score across the three models for each level of gamma for both arrays,
# creating two arrays of length 6, and return a tuple with the two arrays.

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    global X_subset, y_subset

    param_range = np.logspace(-4,1,6)

    train_scores, test_scores = validation_curve(SVC(kernel='rbf', C=1, random_state=0),
    X_subset, y_subset, param_name = 'gamma', param_range = param_range, scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    return (train_scores_mean, test_scores_mean)


# Question 7
# Based on the scores from question 6,
# what gamma value corresponds to a model that is underfitting
# (and has the worst test set accuracy)?
# What gamma value corresponds to a model that is overfitting
# (and has the worst test set accuracy)?
# What choice of gamma would be the best choice for a model
# with good generalization performance on this dataset
# (high accuracy on both training and test set)?
#
# Hint: Try plotting the scores from question 6 to visualize the relationship between gamma
# and accuracy. Remember to comment out the import matplotlib line before submission.
#
# This function should return one tuple with the degree values in this order:
# (Underfitting, Overfitting, Good_Generalization) Please note there is only one correct solution.



def answer_seven(x):


    train_scores_mean, test_scores_mean = x


    plt.title('Validation curve with SVM')
    plt.xlabel('$\gamma$ gamma')
    plt.ylabel('accuracy')

    plt.semilogx(np.logspace(-4,1,6), train_scores_mean, label='Training Scores',color='darkorange', lw=2)
    plt.semilogx(np.logspace(-4,1,6), test_scores_mean,label='Test Scores', color='navy',lw=2)

    plt.fill_between(np.logspace(-4,1,6), train_scores_mean, test_scores_mean,
    train_scores_mean - test_scores_mean, alpha=0.2, color='darkorange', lw=2)

    plt.legend(loc='best')
    plt.show()


    #return is based on visual from chart

    return (0.001, 10, 0.1)



answer_seven(answer_six())







#
