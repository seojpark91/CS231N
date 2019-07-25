import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        # skip for the true class to only loop over incorrect classes
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i, :]
        # for y==i, add negative gradient
        dW[:, y[i]] -= X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss. 
  # Multiply 0.5 so that gradients of the loss come out without any constant in front
  loss += 0.5 * reg * np.sum(W * W)
  # gradient of the above loss
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)

  # select indexes to extract correct label scores
  correct_label_scores_idxes = (range(scores.shape[0]), y)

  # select correct label scores from scores matrix
  correct_label_scores = scores[correct_label_scores_idxes]

  # subtract correct label scores from scores matrix and add 1 for threshold
  margins = scores - correct_label_scores.reshape(-1,1) + 1

  # zero out correct label scores so that we can add losses
  margins[correct_label_scores_idxes] = 0

  # since SVM loss is a hinge loss, 0 out all margins that are less than 0 
  idxes_to_zero_out = np.nonzero(margins < 0)
  margins[idxes_to_zero_out] = 0

  # sum over all dimensions to get loss
  loss = np.sum(margins)

  # divide by the number of training data
  loss /= num_train

  # Add regularization to the loss. 
  # Multiply 0.5 so that gradients of the loss come out without any constant in front
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # as we already calculated margins less than 0, only change margins that are greater 
  # than 0 to 1, which is number of classes - 1
  margins[margins > 0] = 1
  correct_label_gradient_dir = margins.sum(axis=1) * -1
  margins[correct_label_scores_idxes] = correct_label_gradient_dir

  dW = X.T.dot(margins)
  dW /= num_train
  
  # add regularization to the gradient
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
