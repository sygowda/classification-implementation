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

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            delta = (np.where(((np.dot(w, X.T) + b) * y) <= 0, 1, 0)) * y
            w += (step_size * np.dot(delta, X) / N)
            b += (step_size * np.sum(delta) / N)
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            z = (np.matmul(X, w) + b) * y
            w_cal = np.matmul(X.T, sigmoid(-z) * y)
            w += step_size * (w_cal / N)
            b_cal = np.matmul(sigmoid(-z), y)
            b += step_size * (b_cal / N)
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
    value = 1 / (1 + np.exp(-z))
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
        z = np.matmul(X, w) + b
        preds = (z > 0).astype(int)
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.zeros(N)
        z = sigmoid(np.matmul(X, w) + b)
        preds = (z > 0.5).astype(int)
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
    X = np.insert(X, X.shape[1], X.shape[0] * [1], axis=1)
    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    y_labels = list(np.unique(y))
    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        for iter in range(max_iterations):
            y_prime = np.zeros(C)
            idx = np.random.choice(N)
            idx_y = y_labels.index(y[idx])
            y_prime[idx_y] = 1

            intermediate = np.dot(w, X[idx])
            max_exp = np.max(intermediate)
            intermediate = intermediate - max_exp
            exp = np.exp(intermediate)
            sum_value = np.sum(exp)
            exp = exp / sum_value
            exp = exp - y_prime
            w = w - step_size * (exp[:, None]) * (np.transpose((X[idx])[:, None]))

        b = w[:, -1]
        w = w[:, :-1]

    ############################################

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        y_prime = np.zeros((C, N))
        for i in range(N):
            index = y_labels.index(y[i])
            y_prime[index][i] = 1
        for iter in range(max_iterations):
            exp = np.exp(np.dot(w, np.transpose(X)))
            sum_value = np.sum(exp, axis=0)
            exp = exp / sum_value
            exp = exp - y_prime
            w = w - float(step_size / N) * np.dot(exp, X)

        b = w[:, -1]
        w = w[:, :-1]
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."

    # assert w.shape == (C, D)
    # assert b.shape == (C,)

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
    preds = np.argmax(np.matmul(X, w.T) + b, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds




