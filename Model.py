#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score
import warnings
import pickle 
# import matplotlib.pyplot as plt
# %matplotlib inline


# In[ ]:


df = pd.read_csv("HR_comma_sep.csv")


# In[ ]:


df1 = df.drop([ 'last_evaluation', 'Department', 'promotion_last_5years'],axis='columns')


# In[ ]:


n_sal = preprocessing.LabelEncoder()
df1['salary'] = n_sal.fit_transform(df1['salary']) 


# In[ ]:


X = df1[['satisfaction_level', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'salary']].values
Y = df1['left'].values


# In[ ]:



from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3)


# In[ ]:


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


Ks = 40
mean_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(Y_test, yhat)
# print( "The Highest accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:


m1 = LogisticRegression(max_iter = 500)
m2 = LinearRegression()
m3 = SVC()
m4 = KNeighborsClassifier(n_neighbors = 1)


# In[ ]:


Model1 = m1.fit(X_train, Y_train)
Model2 = m2.fit(X_train, Y_train)
Model3 = m3.fit(X_train, Y_train)
Model4 = m4.fit(X_train, Y_train)


# In[ ]:


p1 = m1.predict(X_test)
p2 = m2.predict(X_test)
p3 = m3.predict(X_test)
p4 = m4.predict(X_test)

# print("Accuracy of Logistic Regression :", metrics.accuracy_score(Y_test,p1))
# print("Linear Regression R2-score: %.2f" % r2_score(p2 , Y_test))
# print("Accuracy of svc :", metrics.accuracy_score(Y_test,p3))
# print("Accuracy of KNN :", metrics.accuracy_score(Y_test,p4))

a1 = metrics.accuracy_score(Y_test, p1)
a2 = r2_score(p2 , Y_test)
a3 = metrics.accuracy_score(Y_test, p3)
a4 = metrics.accuracy_score(Y_test, p4)


# In[ ]:


pickle.dump(Model1,open('log_model.pkl','wb'))
pickle.dump(Model2,open('lin_model.pkl','wb'))
pickle.dump(Model3,open('svc_model.pkl','wb'))
pickle.dump(Model4,open('knn_model.pkl','wb'))


# In[ ]:


Y_test


# In[ ]:


# plt.scatter(p2,Y_test, color='blue')
# plt.plot(X_train, m2.coef_*X_train + m2.intercept_, color='red') #here the regression line is plotted y= (slope*x) + intercept
# # plt.xlabel("Engine size")
# # plt.ylabel("Emission")
# plt.show()


# In[ ]:


# sns.pairplot(df1,hue='left')
# warnings.filterwarnings("ignore")


# In[ ]:


df1.to_csv('plot.csv')


# In[ ]:




