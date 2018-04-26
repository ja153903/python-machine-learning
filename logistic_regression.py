import numpy as np 

class LogisticRegressionGD(object):
  """Logistic regression classifier using gradient descent.
  
  Parameters
  -----------
  eta: float
    Learning rate (between 0.0 and 1.0)
  n_iter: int
    Passes over the training dataset
  random_state: int
    Random number generator seed for random weight initialization

  Attributes
  ----------
  w_: 1d-array
    weights after fitting
  cost_: list
    sum of squares cost function value in each epoch.
  """

  def __init__(self, eta=0.05, n_iter=100, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state
  
  def fit(self, X, y):
    """Fit training data

    Parameters
    ----------
    X: {array-like}, shape=[n_samples, n_features]
      training vectors, where n_samples is the number of samples
      and n_features is the number of features
    y: array-like, shape = [n_samples]
      target values
    
    Returns
    -------
    self: object
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0,
                          scale=0.01,
                          size=1+X.shape[1])
    self.cost_ = []

    for i in range(self.n_iter):
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output)
      self.w_[1:] += self.eta * X.T.dot(errors)
      self.w_[0] += self.eta * errors.sum()
      cost = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1 - output))))
      self.cost_.append(cost)
    
    return self
  
  def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[1:]) + self.w_[0]
  
  def activation(self, z):
    """Compute logistic sigmoid activation"""
    return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.net_input(X) >= 0, 1, 0)
