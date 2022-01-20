import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
import typing_extensions


def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    y_pred = np.dot(X, w)
    err = np.sum(np.square(y_pred - y)) / len(y)
    #####################################################

    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  X_trans = X.transpose()
  XtX = np.linalg.inv(np.dot(X_trans, X))
  w = XtX.dot(X_trans).dot(y)
  #####################################################
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
    X_trans = X.transpose()
    XtX = np.dot(X_trans, X)
    len_XtX = len(XtX)
    inverse = np.linalg.inv(XtX + lambd * np.eye(len_XtX))
    w = inverse.dot(X_trans).dot(y)

  #####################################################
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    min_mae = 1
    bestlambda = pow(2, -14)
    for i in range (-14, 1):
        alambda = pow(2, i)
        w = regularized_linear_regression(Xtrain, ytrain, alambda)
        mae = mean_square_error(w, Xval, yval)
        if mae < min_mae:
            min_mae = mae
            bestlambda = alambda
    #####################################################
    return bestlambda


###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################
    Xpow1 = np.array(X)
    mapped_X = np.array(X)
    for i in range(0, p - 1):
        mapped_X = mapped_X * Xpow1
        for col in range(mapped_X.shape[1]):
            X = np.insert(X, X.shape[1], mapped_X[:, col], axis=1)
    return X


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

