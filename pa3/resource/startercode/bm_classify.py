import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent
    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    new_y = np.where(y == 0, -1, 1)

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0

        for i in range(max_iterations):
            prod_value = new_y * (X.dot(w) + b)
            prod_value = np.where(prod_value <= 0, 1, 0)
            prod_value = prod_value * new_y / N  # average gradient
            w += step_size * (np.transpose(prod_value).dot(X))
            b += step_size * np.sum(prod_value)
            # print(w, b)

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            prod_value = new_y * (X.dot(w) + b)
            sgm = sigmoid(prod_value)
            prod_value = sgm * new_y * np.exp(-prod_value) / N  # average gradient
            w += step_size * (np.transpose(prod_value).dot(X))
            b += step_size * np.sum(prod_value)
        ############################################

    else:
        raise Exception("Loss Function is undefined.")

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #

    value = np.reciprocal(1 + np.exp(-z))
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":

        preds = np.zeros(N)
        preds = X.dot(w) + b
        preds = np.where(preds > 0, 1, 0)

    elif loss == "logistic":

        preds = np.zeros(N)
        prod_value = X.dot(w) + b
        preds = sigmoid(prod_value)
        preds = np.where(preds > 0.5, 1, 0)

    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent
    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        # print(N, D, C, "sizes")
        for i in range(max_iterations):
            idx = np.random.choice(N)
            x_n = X[idx]
            y_n = y[idx]
            softmax_val = softmax_function(x_n, w, b, gd_type)
            print(softmax_val.shape)
            # print(softmax_val.shape, "SGD SOFT MAX SHAPE")
            softmax_val[y_n] -= 1
            w -= step_size * softmax_val.reshape(C, 1).dot(x_n.reshape(1, D))
            b -= step_size * softmax_val
        ############################################

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)

        for i in range(max_iterations):
            softmax_val = softmax_function(X, w, b, gd_type)
            # softmax_val = np.sum(softmax_val, axis=0)
            labels_onehot = np.zeros([N, C])
            labels_onehot[np.arange(N), y] = 1.0
            softmax_val -= labels_onehot
            w -= step_size * softmax_val.T.dot(X) / N
            # print(w.shape)
            b -= step_size * np.sum(softmax_val, axis=0) / N

        ############################################

    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def softmax_function(X, w, b, gd_type):
    power_val = X.dot(w.T) + b
    power_val -= power_val.max()
    exp = np.exp(power_val)
    if gd_type == "sgd":
        result = exp / np.sum(exp)
    elif gd_type == "gd":
        result = exp / np.sum(exp, axis=1, keepdims=True)

    return result


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    preds = np.zeros(N)
    preds = X.dot(w.T) + b
    # print(preds.shape)
    preds = np.argmax(preds, axis=1)

    assert preds.shape == (N,)
    return preds
