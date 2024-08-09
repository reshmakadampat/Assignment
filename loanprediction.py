from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app=Flask(__name__)
with open('loanmodel.pkl','rb') as model_files:
  loanmodel=pickle.load(model_files)

@app.route('/')
def home():
  return render_template('screen.html')

@app.route('/predict' ,methods=['POST'])
def model():
  Loan_ID=request.form['Loan_ID']
  Gender=request.form['Gender']
  Married=request.form['Married']
  Dependents=request.form['Dependents']
  Education=request.form['Education']
  Self_Employed=request.form['Self_Employed']
  ApplicantIncome=int(request.form['ApplicantIncome'])
  CoapplicantIncome=float(request.form['CoapplicantIncome'])
  LoanAmount=float(request.form['LoanAmount'])
  Loan_Amount_Term=float(request.form['Loan_Amount_Term'])
  Credit_History=float(request.form['Credit_History'])
  Property_Area=request.form['Property_Area']

  data = pd.DataFrame({
        'Gender': [Gender],
        'Married': [Married],
        'Dependents':[Dependents],
        'Education': [Education],
        'Self_Employed': [Self_Employed],
        'Property_Area': [Property_Area]
    })
  encoded_data = pd.get_dummies(data, dtype=int)
  expected_columns = [
    "Gender_Female", "Gender_Male", 
    "Married_No", "Married_Yes", 
    "Dependents_0", "Dependents_1", "Dependents_2", "Dependents_3+", 
    "Education_Graduate", "Education_Not Graduate", 
    "Self_Employed_No", "Self_Employed_Yes", 
    "Property_Area_Rural", "Property_Area_Semiurban", "Property_Area_Urban"
  ]


  for col in expected_columns:
    if col not in encoded_data.columns:
        encoded_data[col] = 0
  encoded_data = encoded_data[expected_columns]
  scaling=np.array([[ApplicantIncome,
                      CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History]])
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  scaling=scaler.fit_transform(scaling)
  scaled=pd.DataFrame(scaling,columns=['ApplicantIncome',
                      'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History'])
  features=pd.concat([scaled,encoded_data],axis=1)

  prediction=loanmodel.predict(features)
  status='Approved' if prediction[0] == 1 else 'Not Approved'

  return render_template('ans.html',prediction_result=status)

if __name__=='__main__':
  app.run(debug=True)