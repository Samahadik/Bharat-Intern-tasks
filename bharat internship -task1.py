#!/usr/bin/env python
# coding: utf-8

# # importing the libraries
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# 

# In[2]:


import os


# In[4]:


os.chdir("Documents")


# In[5]:


os.getcwd()


# # Reading the dataset

# In[6]:


tg=pd.read_csv('NSE-TATAGLOBAL11.csv')


# In[7]:


tg.head()


# In[8]:


#displaying the basic statistics about data
tg.describe()


# In[9]:


NAN=[(c,tg[c].isnull().mean()*100)for c in tg]
NAN=pd.DataFrame(NAN,columns=['column_name','percentage'])
NAN


# Sorting the data
# checking for null values

# In[10]:


srt=tg.sort_values(by="Date")
srt.head()


# In[11]:


srt.reset_index(inplace=True)
srt.head()


# # Data Visualization
Plotting the graph for the Date and Close
# In[12]:


plt.figure(figsize=(10,7))
plt.plot(srt['Date'],srt['Close'])


# In[13]:


close_srt=srt['Close']
close_srt


# # feature scaling
# (MinMaxScaler)

# In[14]:


scaler=MinMaxScaler(feature_range=(0,1))
close_srt=scaler.fit_transform(np.array(close_srt).reshape(-1,1))
close_srt


# # Splitting the data set

# In[15]:


train_size=int(len(close_srt)*0.7)
test_size=len(close_srt)-train_size
train_data,test_data=close_srt[0:train_size,:],close_srt[train_size:len(close_srt),:1]


# In[16]:


train_data.shape


# In[17]:


test_data.shape


# convert an array of values into a dataset matrix

# In[18]:


def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX),np.array(dataY)


# Reshaping the dataset

# In[19]:


time_step=100
X_train,y_train=create_dataset(train_data,time_step)
X_test,y_test=create_dataset(test_data,time_step)


# In[20]:


print(X_train.shape),print(y_train.shape)


# In[21]:


print(X_test.shape),print(y_test.shape)


# In[22]:


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[23]:


X_train


# In[25]:


pip install tensorflow


# In[26]:


import tensorflow as tf


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# # creating LSTM model

# In[30]:


#Creating the LSTM Model
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[31]:


model.summary()


# # Prediction and Checking Performance

# In[32]:


model.fit(X_train,y_train,validation_split=0.1,epochs=60,batch_size=64,verbose=1)


# In[33]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[34]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# Calculating the Performance

# In[35]:


import math
from sklearn.metrics import mean_squared_error


# In[36]:


math.sqrt(mean_squared_error(y_train,train_predict))


# In[37]:


math.sqrt(mean_squared_error(y_test,test_predict))


# Plotting the graph with predicted train data,test data with actual data

# In[38]:


look_back=100
#shift train prediction for plotting
trainPredictPlot=np.empty_like(close_srt)
trainPredictPlot[:, :]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
#shift test predictions for plotting
testPredictPlot=np.empty_like(close_srt)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(close_srt)-1,:]=test_predict
#plot baseline and predictions
plt.figure(figsize=(10,7))
plt.plot(scaler.inverse_transform(close_srt))
plt.plot(testPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# Prediction for the next 30 days

# In[39]:


len(test_data)


# In[40]:


pred_input=test_data[271:].reshape(1,-1)


# In[41]:


temp_input=list(pred_input)
temp_input=temp_input[0].tolist()


# In[42]:


temp_input


# In[44]:


lst_output=[] #predicted 30 days output
n_steps=100
i=0
while(i<30): #for 30 days-change according to number of days you want
    if(len(temp_input)>100):
        #print(temp_input)
        pred_input=np.array(temp_input[1:])#for last 100 days
        print("{} day input {}".format(i,pred_input))
        pred_input=pred_input.reshape(1,-1)
        pred_input=pred_input.reshape((1,n_steps,1))
        #print(X_input)
        yhat=model.predict(pred_input,verbose=0)
        print("{} day output{}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        pred_input=pred_input.reshape(1,n_steps,1)
        yhat=model.predict(pred_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
        
print(lst_output)


# Plotting the lasst 30 days closing Price

# In[45]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[46]:


len(close_srt)


# In[50]:


plt.figure(figsize=(10,7))
plt.plot(day_new,scaler.inverse_transform(close_srt[1135:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# #### Appending in a list i done to make continious graph

# In[51]:


df3=close_srt.tolist()
#adding specified list of predicted 30 days output to the end of last 100days output
df3.extend(lst_output)


# In[52]:


print(len(df3))


# In[53]:


plt.figure(figsize=(10,7))
plt.plot(df3[1135:]) #latest 100 days output


# In[ ]:


df3=scaler.inverse_transform(df3).tolist()  #undoing scaling of df


# #### Plotting the graph with predicted 30 days output

# In[54]:


plt.figure(figsize=(10,7))
plt.plot(df3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




