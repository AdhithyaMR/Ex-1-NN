<H3>ENTER YOUR NAME:  Adhithya M R
<H3>ENTER YOUR REGISTER NO.  212222240002
<H3>EX. NO. 1
<H3>DATE:  27/02/2024
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))
```



## OUTPUT:
### DATASET
![308785272-2fb85e34-a1d3-4f37-9ddd-08158ecc16d5](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/4d6d7cb4-2ccc-4c88-aad7-6c722210d445)
### XVALUES:
![308785415-78897b56-4e80-4ccb-afea-2686b900fe40](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/0f2b6a2b-f3ae-41fb-afe4-925d960e0107)
### Y VALUES:

![308785498-c91a7483-1090-4fd7-ac9b-1aed66d6a66b](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/6f282443-ee55-42b0-8fe7-d600af739801)
### NULL:
![308785592-1523c24e-e575-4d67-b093-e248ac06145d](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/6f537d8a-9258-4c8d-b2a7-2c72c1fe8c66)
### DUPLICATE:
![308785690-849ec8ab-63f4-4876-987d-077cbae6f320](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/13a56f90-e442-41f4-a402-d6f82f672459)
### DESCRIBE:
![308785813-775ae6df-165a-42f5-a820-08a95b67182a](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/6c8612cb-be0b-4d43-a51a-c03e0a531f7c)
### DATASER AFTER DROPPING:
![308785989-0763d987-7c2c-4feb-b57a-f18324ddaed1](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/ceb24f4e-b788-4cd3-a238-438197d099e4)
### NORMALIZE DATASET:
![308786094-31953775-7000-4fe3-8b83-f8ecd8ce3aae](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/ffe9ab47-472b-47ea-8a54-e53abe936b37)
### X TRAIN:
![308786228-91623ab4-37a9-4258-b0c8-fc2a1b8bdfb8](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/528f87ee-2ec9-46b6-8ee7-3f2b6f0eea22)
### X TEST:
![308786339-dcfdc6fa-1093-4bee-9e62-84bd2d5fec68](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/fc9dc804-90b7-4cf9-bf97-eeb5e855f1fb)
### LENGTH:
![308786427-56ccbcdf-1786-49f8-b4c1-071f57e4f8c5](https://github.com/AdhithyaMR/Ex-1-NN/assets/118834761/c7bd58b9-5865-434e-9df5-e1ac242eb5ca)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


