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
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_dim = W.shape[0]
  
  for i in xrange(num_train):
    scores = X[i].dot(W)
   # to avoid numeric instability, subtract max value from scores
    exp_scores = np.exp(scores - np.max(scores))
    prob_scores = exp_scores / np.sum(exp_scores)

    for d in xrange(num_dim):
      for k in xrange(num_classes):
        if k == y[i]:
          # gradient of loss with respect to weight
           dW[d, k] += (prob_scores[k]-1) * X.T[d, i]
        else:
          dW[d, k] += prob_scores[k] * X.T[d, i]

    loss += - np.log(prob_scores[y[i]])

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  scores = X.dot(W)
  exp_scores = np.exp(scores - np.max(scores))
  # keepdims parameter to match dimensions
  prob_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  correct_log_probs = - np.log(prob_scores[range(num_train), y])

  # calculate loss
  loss = np.sum(correct_log_probs)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  # calculate gradient
  
  grad_scores = prob_scores
  # if i==j, subtract 1 from the probability scores, then get the gradients
  grad_scores[range(num_train), y] -=1
  dW = np.dot(X.T, grad_scores)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

