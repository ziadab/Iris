
# coding: utf-8

# In[17]:


from sklearn.datasets import load_iris
from sklearn import tree


# In[18]:


iris = load_iris()


# In[19]:


#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])


# In[21]:


import numpy as np
test_idx = [0,50,100]

#training data
train_taget = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#print(train_data)


# In[8]:


clf_train = tree.DecisionTreeClassifier()
clf_train = clf_train.fit(train_data ,train_taget)


# In[9]:


#print(test_target)
#print(clf_train.predict(test_data))


# In[10]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data,iris.target)


# In[15]:


print("This Tools is Just to predict if The Isis Flower is Setosa, Versicolor or Virginica\n")
print("The main Features are sepal length(cm), sepal width(cm), petal length(cm) and petal width(cm)\nYou can mesure them using a ruler\n")

sepal_length = float(input("Sepal length(cm) : "))
sepal_width = float(input("Sepal width(cm) : "))

petal_length = float(input("Petal length(cm) : "))
petal_width = float(input("Petal width(cm) : "))


# In[23]:


iris_predict = clf.predict([[sepal_length,sepal_width,petal_length,petal_width]])


# In[24]:


#print(iris_predict)


# In[25]:


if int(iris_predict) == 0:
    print("\nYour Iris is 'Setosa'")
elif int(iris_predict) == 1:
    print("\nYour Iris is 'Versicolor'")
else :
    print("\nYOur Iris is 'Virginica'")
    

