import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta


def do_lsq_inference(X, y, beta):
    """
    Perform inference on the least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :param beta: Estimated coefficient vector for the linear regression
    :return: Prediction for y, MSE
    """
    # y_pred
    print(f'shapes: X {X.shape}, beta {beta.shape}')
    y_pred = np.matmul(X, beta).squeeze()

    # mse
    mse = np.mean((y - y_pred) ** 2)

    return y_pred, mse