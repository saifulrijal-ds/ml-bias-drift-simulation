# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("lgbm_test_api")

# Create input/output pydantic models
input_model = create_model("lgbm_test_api_input", **{'Age': 30, 'Gender': 'Male', 'Location': 'Java/Urban', 'Education_Level': 'University', 'Occupation_Type': 'Entrepreneur', 'Marital_Status': 'Married', 'Number_of_Dependents': 3, 'Years_at_Address': 12, 'Years_at_Employment': 9, 'Monthly_Income': 15000789, 'Loan_Amount': 30000000, 'Loan_Term': 36, 'Monthly_Payment': 916666.6875, 'Payment_to_Income_Ratio': 0.06110789626836777, 'Existing_Debt_Amount': 13591915, 'Prior_Loans': 'Yes', 'Payment_History': 'Good', 'Bank_Account_Type': 'Premium', 'Credit_Card': 'Yes'})
output_model = create_model("lgbm_test_api_output", prediction='Fully Paid')


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
