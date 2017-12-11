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
  num_train = X.shape[0]
  num_class = W.shape[1]
  loss = 0.0
  dW = np.zeros(W.shape)
  margins = []
  
  for i in range(num_train):
    scores = np.dot(X[i], W)
    margins = np.maximum(0, scores - scores[y[i]] + 1)
    margins[y[i]] = 0
    loss += np.sum(margins)

    for j in range(num_class):
      if j == y[i]:
        continue;
      if margins[j] > 0:
        dW[:, j] += X[i]
        dW[:,y[i]] -= X[i] 
       
  loss /= num_train
  dW /= num_train

  loss += reg*np.sum(W**2)
  dW += 2*reg*W




  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  
  loss = 0.0
  dW=np.zeros(W.shape)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = np.dot(X, W)
  margins =np.maximum(0, scores - scores[range(num_train), y].reshape(num_train,-1) + 1) 
  margins[range(num_train), y] = 0
  loss = np.sum(margins)/num_train + reg*np.sum(W**2)

  X_mask = np.zeros(margins.shape)
  X_mask[margins > 0] = 1     
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts

  dW = X.T.dot(X_mask)

  dW /= num_train
  dW += reg * W            

  return loss, dW
