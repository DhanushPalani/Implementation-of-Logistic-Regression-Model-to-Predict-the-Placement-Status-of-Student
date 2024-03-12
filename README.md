# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
  1.Import the standard libraries.
  2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively
  3.Import LabelEncoder and encode the dataset.
  4. Import LogisticRegression from sklearn and apply the model on the dataset.
  5. Predict the values of array.
  6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
  7. Apply new unknown values
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DHANUSH P
RegisterNumber:  212222040034
*/

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:\n",confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("\nClassification Report:\n",classification_report1)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
### DATA:
![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/349bdf15-803e-4aa9-ba3b-365ea872c847)

### ENCODED DATA:

![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/54a1182b-abac-4f4a-b302-c79848070922)
### NULL FUNCTION:
![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/bf6ed20e-f120-452a-9c32-4133515d34f7)

### DATA DUPLICATE:
![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/72f5b293-5308-40b8-922c-49e4c65bf586)

### ACCURACY:
![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/7e70d2b6-6797-4b01-81b9-06af7372b636)

### CONFUSION MATRIX:
![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/169cdc2a-5db5-4ce4-84e4-38e9d65104c4)

### CONFUSION REPORT:
![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/972d8f70-fc6d-4f07-b72b-bb092077f2c6)

### PREDICTION OF LR:
![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/0026c994-535c-4294-a8ae-3f2bdb231775)

### GRAPH:
![image](https://github.com/DhanushPalani/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121594640/58ff2b30-19fa-432d-82ae-e7225b82cfbd)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
