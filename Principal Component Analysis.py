#!/usr/bin/env python
# coding: utf-8

# In[31]:


x=[2,3,5,7,9]
y=[1,4,0,6,2]
leng=len(x)
x_mean=sum(x)/leng
y_mean=sum(y)/leng
cov_xx=0
cov_yy=0
cov_xy=0
XX=[]
YY=[]
for i in range(leng):
    cov_xx += (x[i]-x_mean)**2
    cov_yy += (y[i]-y_mean)**2
    cov_xy += (x[i]-x_mean)*(y[i]-y_mean)
    XX.append(x[i]-x_mean)
    YY.append(y[i]-y_mean)
for i in range(leng):
    print(XX[i],"     ",YY[i])
print()
cov_xx/=(leng-1)
cov_yy/=(leng-1)
cov_xy/=(leng-1)
covar=[]
covar.append([cov_xx,-1*cov_xy])
covar.append([-1*cov_xy,cov_yy])
print("Covariencs:")
for i in covar:
    print(i)


# In[20]:


import numpy as np
from numpy.linalg import eig
eigen_values,eigen_vectors=eig(covar)
print(eigen_values)
print(eigen_vectors)


# In[21]:


sorted_indices = sorted(range(len(eigen_values)), key=lambda i: eigen_values[i], reverse=True)
eigenvalues = [eigen_values[i] for i in sorted_indices]
eigenvectors = [eigen_vectors[:, i] for i in sorted_indices]

k = 2
selected_eigenvectors = eigenvectors[:k]
print("Eigenvalues:")
print(eigenvalues)
print("\nSelected Eigenvectors:")
for vector in selected_eigenvectors:
    print(vector)


# In[4]:


import matplotlib.pyplot as plt
plt.subplot(1, 1, 1)
plt.scatter(x, y)
plt.title('Original Data')
plt.xlabel('x')
plt.ylabel('y')


# In[27]:


XX_YY=[]
for i in range(leng):
    XX_YY.append([XX[i],YY[i]])
transformed_data = np.dot(XX_YY, selected_eigenvectors)
print(transformed_data)


# In[28]:


plt.subplot(1, 1, 1)
plt.scatter(transformed_data[:,0], transformed_data[:,1])
plt.title('Transformed Data')
plt.xlabel('x')
plt.ylabel('y')


# In[ ]:




