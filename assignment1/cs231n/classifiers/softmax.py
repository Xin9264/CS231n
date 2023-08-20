from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train, num_dim = X.shape  
    num_class = W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        labels = np.dot(X[i], W)
        probability = np.exp(labels) / np.sum(np.exp(labels), keepdims=True)
        real_label = y[i]
        loss += -np.log(probability[real_label])
        probability[real_label] -= 1
        # dW[i] += X[i][np.newaxis, :].dot(probability[:, np.newaxis])
        dW += np.dot(X[i][:, np.newaxis], probability[np.newaxis, :])
    loss += reg * np.sum(W**2)
    loss = loss / num_train
    dW = dW / num_train
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train, num_class = X.shape
    scores = np.dot(X, W)
    probability = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    real_prob = probability[range(num_train), y]
    loss = np.mean(-np.log(real_prob))
    loss += reg * np.sum(W**2)
    probability[range(num_train), y] -= 1
    dW = X.T.dot(probability)
    dW += 2 * reg * W
    dW = dW / num_train
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
