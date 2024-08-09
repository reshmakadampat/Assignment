import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv('train_ctrUa4K.csv')
missing_val_col_num=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History']
for col in missing_val_col_num:
       train[col]=train[col].fillna(train[col].median())
missing_val_col_obj=['Gender','Married','Dependents','Self_Employed']
for col in missing_val_col_obj:
       train[col]=train[col].fillna(train[col].mode()[0])
large_unique_col_train=['ApplicantIncome','CoapplicantIncome','LoanAmount']
for col in large_unique_col_train:
    q1 = train[col].quantile(0.25)
    q3 = train[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers= ((train[col] < lower_bound) | (train[col] > upper_bound))
    train[col] = train[col].where(~outliers, np.median(train[col]))
train1=train.copy()
train1.drop(columns='Loan_ID',inplace=True)
data2=train1[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Property_Area']]
data2=pd.get_dummies(data2)
for i in data2:
  data2[i]=data2[i].astype('int64')
train1=train1.drop(columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'Property_Area'])
train1 = pd.concat([train1, data2], axis=1)
status_mapping = {
    'N': 0,
    'Y': 1,
}
train1['Loan_Status'] = train1['Loan_Status'].map(status_mapping)
x=train1.drop(columns=['Loan_Status'])
y=train1['Loan_Status']
x1=x.copy()
x1.drop(['Gender_Female', 'Gender_Male', 'Married_No', 'Married_Yes',
       'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
       'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_No',
       'Self_Employed_Yes', 'Property_Area_Rural', 'Property_Area_Semiurban',
       'Property_Area_Urban'],axis=1,inplace=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x1=scaler.fit_transform(x1)
x1=pd.DataFrame(x1,columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History'])
x2=x.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History'],axis=1)
x=pd.concat([x1,x2],axis=1)
from sklearn.ensemble import RandomForestClassifier
rf_clf=RandomForestClassifier(random_state=42,n_estimators=200,
    max_depth=10,
    max_features='sqrt',
    bootstrap=True)
rf_clf.fit(x,y)
import pickle
with open('loanmodel.pkl','wb') as model_file:
  pickle.dump(rf_clf,model_file)