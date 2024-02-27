
import numpy as np


class LogLoss:
    @staticmethod
    def get_loss(y, y_hat):
        loss = -(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
        return loss

    @staticmethod
    def get_gradient(y, y_hat):
        epsilon = 1e-8  # small constant to avoid division by zero
        gradient = -(y / (y_hat + epsilon)) + ((1 - y) / ((1 - y_hat) + epsilon))
        return gradient

    @staticmethod
    def get_hessian(y, y_hat):
        epsilon = 1e-8  # sm
        hessian = (y / ((y_hat ** 2) + epsilon)) + ((1 - y) / ((1 - y_hat) ** 2) + epsilon)
        return hessian


class MSE:
    @staticmethod
    def get_loss(y, y_hat):
        loss = np.mean((y - y_hat) ** 2)
        return loss

    @staticmethod
    def get_gradient(y, y_hat):
        gradient = -2 * (y - y_hat)
        return gradient

    @staticmethod
    def get_hessian(y, y_hat):
        # Since the second derivative of the MSE loss function is a constant,
        # we simply return a matrix of 2's with the same shape as y.
        hessian = 2 * np.ones_like(y)
        return hessian