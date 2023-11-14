#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir("Documents")


# In[5]:


os.getcwd()


# In[6]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[7]:


train.head()


# In[8]:


test.head()


# In[9]:


print(train.shape)
print(test.shape)


# In[10]:


train.info()


# In[11]:


test.info()


# In[12]:


train.drop(columns=['Cabin'],inplace=True)
test.drop(columns=['Cabin'],inplace=True)


# In[13]:


train.isnull().sum()


# In[14]:


test.isnull().sum()


# In[15]:


train['Embarked'].value_counts()


# In[16]:


train['Embarked'].fillna('S',inplace=True)


# In[17]:


test['Fare'].fillna(test['Fare'].mean(),inplace=True)


# In[18]:


train_age=np.random.randint(train['Age'].mean()-train['Age'].std(),train['Age'].mean()+train['Age'].std(),177)


# In[19]:


test_age=np.random.randint(test['Age'].mean()-test['Age'].std(),test['Age'].mean()+test['Age'].std(),86)


# In[20]:


train['Age'][train['Age'].isnull()]=train_age


# In[21]:


train['Age'].isnull().sum()


# In[22]:


test['Age'][test['Age'].isnull()]=test_age


# In[23]:


train['Age'].isnull().sum()


# In[24]:


#EDA


# In[25]:


train.groupby(['Pclass'])['Survived'].mean()


# In[26]:


train.groupby(['Sex'])['Survived'].mean()


# In[27]:


train.groupby(['Embarked'])['Survived'].mean()


# In[28]:


sns.distplot(train['Age'])


# In[29]:


sns.distplot(train['Age'][train['Survived']==0])
sns.distplot(train['Age'][train['Survived']==1])


# In[30]:


#blue curve for not survived people
#red for survived people


# In[31]:


sns.distplot(train['Fare'][train['Survived']==0])
sns.distplot(train['Fare'][train['Survived']==1])


# In[32]:


#there is positive correlation between fare and survival chances
#that is ,as fare ncreases survival chances increases


# In[33]:


train.drop(columns=['Ticket'],inplace=True)
test.drop(columns=['Ticket'],inplace=True)


# In[34]:


train['Family']=train['SibSp']+train['Parch']+1
test['Family']=test['SibSp']+test['Parch']+1


# In[35]:


train['Family'].value_counts()


# In[36]:


train.groupby(['Family'])['Survived'].mean()


# In[37]:


test.head(3)


# In[38]:


def cal(number):
    if number==1:
        return "Alone"
    elif number>1 and number<5:
        return "Medium"
    else:
        return "Large"


# In[39]:


train['Family size']=train['Family'].apply(cal)
test['Family size']=test['Family'].apply(cal)


# In[40]:


test.head(2)


# In[42]:


train.drop(columns=['SibSp','Parch','Family'],inplace=True)
test.drop(columns=['SibSp','Parch','Family'],inplace=True)


# In[43]:


print(train.shape)
print(test.shape)


# In[44]:


PassengerID=test['PassengerId'].values


# In[45]:


train.drop(columns=['Name','PassengerId'],inplace=True)
test.drop(columns=['Name','PassengerId'],inplace=True)


# In[46]:


train.isnull().sum()


# In[47]:


train.shape


# In[49]:


train=pd.get_dummies(train,columns=['Pclass','Sex','Embarked','Family size'],drop_first=True)


# In[50]:


train.shape


# In[51]:


test=pd.get_dummies(test,columns=['Pclass','Sex','Embarked','Family size'],drop_first=True)


# In[52]:


test.shape


# In[53]:


X=train.iloc[:,1:].values
y=train.iloc[:,0].values


# In[59]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[60]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()


# In[61]:


classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


# In[62]:


from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)


# In[ ]:


#we will find whether the person survive or not for test data


# In[63]:


XFinal=test.iloc[:,:].values


# In[64]:


y_final=classifier.predict(XFinal)


# In[65]:


final=pd.DataFrame()


# In[69]:


final['PassengerId']=PassengerID
final['survived']=y_final


# In[70]:


final


# In[ ]:


####survival of person is depends on Pclass,Sex,Age,Fare,Embarked,Family size 


# In[ ]:




