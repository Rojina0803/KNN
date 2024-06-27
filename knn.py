import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

data=pd.read_csv("Classified_Data.csv")
# print(data.head())
#USING KNN
# print(data.columns)
# print(data.info())

# for data normalization
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(data.drop('TARGET CLASS',axis=1))
scaled_features=scaler.transform(data.drop('TARGET CLASS',axis=1))
# print(scaled_features)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(scaled_features,data['TARGET CLASS'],test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

pred=knn.predict(X_test)
# print(pred)

#MODEL EVALUATION
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))

error=[]
for i  in range(1,25):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i=knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.plot