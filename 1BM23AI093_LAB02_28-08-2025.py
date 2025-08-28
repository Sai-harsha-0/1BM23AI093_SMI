#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


import numpy as np

def gradient_descent(X, y, initial_learning_rate=0.01, decay_rate=0.01, n_iterations=1000):
    m = len(y)
    theta = np.random.randn(X.shape[1])
    for iteration in range(n_iterations):
        gradients = (2/m) * X.T.dot(X.dot(theta) - y)
        learning_rate = initial_learning_rate / (1 + decay_rate * iteration)
        theta -= learning_rate * gradients
    return theta

theta_gd = gradient_descent(X, y)
print("\nGradient Descent:")
print(f"Intercept: {theta_gd[0]}, Slope: {theta_gd[1]}")


# In[17]:


import numpy as np

def stochastic_gradient_descent(X, y, learning_rate=0.001, n_iterations=100000):
    m = len(y)
    theta = np.array([3000, -0.1])
    for iteration in range(n_iterations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        theta -= learning_rate * gradients
    return theta

theta_sgd = stochastic_gradient_descent(X, y)
print("\nStochastic Gradient Descent:")
print(f"Intercept: {theta_sgd[0]}, Slope: {theta_sgd[1]}")


# In[ ]:




