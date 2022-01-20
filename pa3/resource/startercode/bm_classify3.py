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

    w = np.insert(w, 0, b)
    y = np.where(y == 0, -1, y)
    X = np.insert(X, 0, 1, axis=1)

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #

        # Gradient = sum(-I(yn * w_T * Xn <= 0) * ynXn)
        # only when (yn * w_T * Xn <= 0) does the example contribute to gradient
        # in every iteration update w with w + step_size * gradient

        for i in range(2):
            mul = np.multiply(y, np.dot(X, w))
            # print(i)
            matched_idx = np.where(mul > 0)
            mis_y = np.delete(y, matched_idx)
            mis_x = np.delete(X, matched_idx, axis=0)
            mis_sum = np.matmul(np.transpose(mis_x), np.transpose(mis_y))
            # print(mis_sum)
            mis_arg = np.divide(mis_sum, N)
            # print(mis_arg)
            step = np.multiply(step_size, mis_arg)
            w = np.add(w, step)

        b = w[0]
        w = w[1:]
        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #

        # Gradient = sum(sigmoid(-yn * (w_T * Xn + b)) * ynXn)
        # in every iteration update w with w + step_size * gradient
        for i in range(max_iterations):
            mul = np.multiply(y, np.dot(X, np.transpose(w)))
            mis_prob = sigmoid(np.multiply(-1, mul))
            mis_sum = np.matmul(np.transpose(X), np.transpose(np.multiply(mis_prob, y)))

            mis_arg = np.divide(mis_sum, N)
            step = np.multiply(step_size, mis_arg)
            w = np.add(w, step)

        b = w[0]
        w = w[1:]
        ############################################

    else:
        raise "Loss Function is undefined."

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
    value = np.power(1 + np.exp(-z), -1)
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
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        pred_y = np.dot(X, np.transpose(w)) + b

        # predict y based on whether pred_y > 0
        for i in range(N):
            if pred_y[i] > 0:
                preds[i] = 1
            else:
                preds[i] = 0
        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        y = np.dot(X, np.transpose(w)) + b

        # predict y based on sigmoid
        for i in range(N):
            if sigmoid(y[i]) > 0.5:
                preds[i] = 1
            else:
                preds[i] = 0
        ############################################

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
    - y: multi-class training labels, an N dimensional numpy array where
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
        for _ in range(max_iterations):
            n = np.random.choice(N)
            # y_n is the label of chosen data
            y_n = y[n]
            X_n = X[n]
            # shape of subtracted_w is C * (D + 1)
            # subtracted_w = w - w[y_n]
            # subtracted_w[y_n] = w[y_n]
            # shape of numer is C * 1
            power = w.dot(X_n) + b
            # numer = np.exp(numer)
            numer = np.exp(power - np.max(power))
            denom = np.sum(numer)
            softmax = np.divide(numer, denom)
            softmax[y_n] -= 1

            step = np.multiply(step_size, np.dot(softmax[:, None], X_n[None, :]))
            w = np.subtract(w, step)
            b = np.subtract(b, np.multiply(step_size, softmax))

        # b = w[:, 0]
        # w = np.delete(w, 0, axis=1)
        ############################################

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        X = np.insert(X, 0, 1, axis=1)
        w = np.insert(w, 0, b, axis=1)

        for i in range(max_iterations):
            numer = np.matmul(w, np.transpose(X))
            x_max = np.tile(np.amax(numer, axis=0), (C, 1))
            numer = np.exp(np.subtract(numer, x_max))
            denom = np.tile(np.sum(numer, axis=0), (C, 1))
            g = np.divide(numer, denom)
            g[y, np.arange(N)] -= 1

            step = np.matmul(g, X) / N
            step = np.multiply(step_size, step)

            w = np.subtract(w, step)

        b = w[:, 0]
        w = np.delete(w, 0, axis=1)

        ############################################

    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


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
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)
    X = np.insert(X, 0, 1, axis=1)
    w = np.insert(w, 0, b, axis=1)
    y = np.matmul(w, np.transpose(X))

    # predict based on max value of class
    preds = np.argmax(y, axis=0)
    ############################################

    assert preds.shape == (N,)
    return preds

