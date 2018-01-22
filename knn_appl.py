import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split


df = pd.read_csv('breast_cancer.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)
y=np.array(df['label'])
x=np.array(df.drop(['label'],1))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)
print(accuracy)

example_data = np.array([[5,1,2,2,2,1,3,1,1],[5,3,1,1,2,1,3,1,1]])
example_data = example_data.reshape(len(example_data),-1)
prediction = clf.predict(example_data)
print(prediction)