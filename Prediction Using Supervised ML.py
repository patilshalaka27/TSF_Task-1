#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Spark Foundation

# # Data Science and Bussiness Analytics Intern

# Author: Shalaka Patil

# # Task 1 : Prediction Using Supervised ML

# In this task we have to predict the percentage score of a student based on the number of hours studied. The task has two variables where the features is the no. of hours studied and the target values is the percentage score. This can be solved using Simple Linear Regression.

# In[1]:


# Import Python Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Impoprting data set
data =pd.read_csv("C:/Users/DELL/OneDrive/Desktop/data.txt")
print("Importing Data Sucesssfully")
data.head(10)


# In[3]:


# Check wheather Data imported successfully or not
print("For this we print first 10 data of Data.set")
data.head(10)
print("You have correctly imported Data set")


# In[4]:


# Plot the graph, for detail analysis of data
data.plot(x='Hours',y='Scores',style='1')
plt.title("Hours Vs Percentage")
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[5]:


data.plot.pie(x='Hours',y='Scores')


# In[6]:


data.plot.scatter(x='Hours',y='Scores')


# In[7]:


data.plot.bar(x='Hours',y='Scores')


# In[8]:


data.sort_values(["Hours"],axis=0,ascending=[True],inplace=True)
data.head(10)
data.plot.bar(x='Hours',y='Scores')


# After plotting Different graph,we have observed that as study hours increases,score is also increase.Which is good sign of correct data.In our daily life,we have observed the same phenomenon.

# In[9]:


# Now we have prepared the data for our model
X=data.iloc[:, :-1].values
y=data.iloc[:, 1].values
#print(X)


# In[10]:


# Now we have divide the data from training and testing the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                         test_size=0.2, random_state=0)


# In[11]:


# Training the Algorithm
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()


#from sklearn.enesemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 1000,random_state= 42)


regressor.fit(X_train, y_train)
print("Training Complete.")


# In[12]:


#Now,Our model is ready it's time to test it.
print(X_test)
print("Predection of Score")
y_pred = regressor.predict(X_test)
print(y_pred)


# In[13]:


#Now checking the Accuracy of Our Model
df = pd.DataFrame ({'Actual': y_test, 'predicted': y_pred})

df


# In[14]:


#Now It's time to prediction with custom Input
hours = [[9.25]]
pred = regressor.predict(hours)
print(pred)


# In[15]:


#Evaluating the Model
from sklearn import metrics
print('Mean Absolute Error:',
     metrics.mean_absolute_error(y_test, y_pred))


# In[16]:


# Training the Algorithm
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000,random_state= 42)


regressor.fit(X_train, y_train)
print("Training Complete.")


# In[17]:


#Now,Our model is ready it's time to test it.
print(X_test)
print("Predection of Score")
y_pred = regressor.predict(X_test)
print(y_pred)


# In[18]:


#Now checking the Accuracy of Our Model
df = pd.DataFrame ({'Actual': y_test, 'predicted': y_pred})

df


# In[19]:


#Now It's time to prediction with custom Input
hours = [[9.25]]
pred = regressor.predict(hours)
print(pred)


# In[20]:


#Evaluating the Model
from sklearn import metrics
print('Mean Absolute Error:',
     metrics.mean_absolute_error(y_test, y_pred))


# #Conclusion
# 1) Linear Regression
#    Mean Absolute Error: 4.621333622532765
# 
# 2) Random Forest
#    Mean Absolute Error: 4.864150000000003
