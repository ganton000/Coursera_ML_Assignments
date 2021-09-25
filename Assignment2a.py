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
X_test=X_test.reshape(-1,1)
X_train = X_train.reshape(-1,1)

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


















#
