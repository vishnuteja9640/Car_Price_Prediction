#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Car Price Prediction Using Python
# Here we are using the dataset from Kaggle and since we can't input our raw data into a machine learning model we have to
# preprocess our data then later split into train and test dataset and then it has to be fed to the model for training.


# In[3]:


# Since this problem is not a classification problem we have to use Linear Regression and LASSO Regression and check accuracy.


# In[217]:


#Now importing the required libraries for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score


# In[218]:


dataframe = pd.read_csv('cardata.csv')
dataframe.shape
# Number of rows and columns for this dataset


# In[219]:


dataframe.head()


# In[220]:


print(dataframe.isnull().sum())
# Checking if the dataframe has any empty datapoints
print(dataframe.info())


# In[221]:


#Checking how many cars are of Petrol and how many card run on Diesel
print((dataframe['Fuel_Type'] == 'Petrol').sum())


# In[222]:


print((dataframe['Fuel_Type'] == 'Diesel').sum())


# In[223]:


(dataframe['Seller_Type'] == 'Dealer').sum()


# In[224]:


(dataframe['Seller_Type'] == 'Individual').sum()


# In[225]:


(dataframe['Transmission'] == 'Manual').sum()


# In[226]:


(dataframe['Transmission'] == 'Automatic').sum()


# In[227]:


# We do the above format if know the name of the variables present in the dataframe
# But, this can also be written with value_counts() method
print(dataframe.Fuel_Type.value_counts())
print(dataframe.Seller_Type.value_counts())
print(dataframe.Transmission.value_counts())


# In[228]:


# As we had Textual Data on Fuel_Type, Seller_Type and Transmission we have to convert this into Numerical Data and then
# we have to fed as input to a Machine Learning model
dataframe.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
# You only have to execute this cell only once 
#dataframe['Fuel_Type'].replace(to_replace = ['Petrol', 'Diesel','CNG'], value =[0,1,2], inplace=True)


# In[229]:


dataframe


# In[230]:


# Same process you have to convert the textual data of Seller_Type and Transmission into Numerical data
dataframe.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
#dataframe['Seller_Type'].replace(to_replace = ['Dealer', 'Indivudual'], value =[0,1], inplace=True)


# In[231]:


dataframe


# In[232]:


dataframe.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)
#dataframe['Transmission'].replace(to_replace = ['Manual', 'Automatic'], value =[0,1], inplace=True)


# In[233]:


dataframe


# In[234]:


# Now splitting our dataset into X and Y
X = dataframe.drop(columns=['Selling_Price','Car_Name'],axis=1)
Y = dataframe['Selling_Price']
# Since Car_Name is also not very important when we are training our model we can remove it. Here our label is Selling_price
# we had stored that variable in Y. 


# In[235]:


print(X)
print(Y)


# In[236]:


# Now splitting the dataset into Training and Testing 
# Here since the dataset has lower rows we are taking the test_size to be 10% of the overall dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,random_state = 2)


# In[237]:


X_train


# In[238]:


# Now training a ML Model (Linear Regression)
model1 = LinearRegression()
model1.fit(X_train,Y_train)


# In[239]:


#Now checking the accuracy of our Linear Regression Model
# Checking the rsquare error for linear regression model
training_accuracy = model.predict(X_train)
rsquare_error = metrics.r2_score(Y_train,training_accuracy)


# In[240]:


print('The rsquare_error of training data is: {}'.format(rsquare_error))
# For Classification problem we use accuracy and for Regression case we use rsquare_error


# In[241]:


# Now plot the actual prices to predicted prices


# In[242]:


plt.scatter(Y_train,training_accuracy)
plt.xlabel('Actual_prices of the Car')
plt.ylabel('Predicted_prices of the Car')
plt.title("Acutal Prices of Car VS Predicted Prices of Car")
plt.show()


# In[243]:


testing_accuracy = model.predict(X_test)
rsquare_error = metrics.r2_score(Y_test,testing_accuracy)
print('The rsquare_error for test data is {}'.format(rsquare_error))


# In[244]:


# Now plotting the graph for testing data
plt.scatter(Y_test,testing_accuracy)
plt.xlabel('Actual_prices of the Car')
plt.ylabel('Predicted_prices of the Car')
plt.title("Acutal Prices of Car VS Predicted Prices of Car")
plt.show()


# In[245]:


# Now check with LASSO Regression
model2 = Lasso()
model2.fit(X_train,Y_train)


# In[246]:


model2_error_score = model2.predict(X_train)
rsquare_error = metrics.r2_score(Y_train,model2_error_score)
print("The R Square error: {}".format(rsquare_error))


# In[247]:


# Now plotting the graph for LASSO model
plt.scatter(Y_train,model2_error_score)
plt.xlabel('Actual_prices of the Car')
plt.ylabel('Predicted_prices of the Car')
plt.title("Acutal Prices of Car with training data VS Predicted Prices of Car with testing data")
plt.show()


# In[248]:


# Now predicting for test data
model2_error_score = model2.predict(X_test)
rsquare_error = metrics.r2_score(Y_test,model2_error_score)
print("The R Square error: {}".format(rsquare_error))


# In[249]:


# Now plotting for testing data
plt.scatter(Y_test,model2_error_score)
plt.xlabel('Actual_prices of the Car')
plt.ylabel('Predicted_prices of the Car')
plt.title("Acutal Prices of Car for test data VS Predicted Prices of Car for test data")
plt.show()


# In[ ]:





# In[ ]:




