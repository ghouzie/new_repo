import joblib
import pandas as pd
from fastapi import FastAPI


app = FastAPI()

## LOAD MODEL ##
pipe = joblib.load('model.pkl')


@app.post("/predict")
def predict(new_data):

    new_data_df = pd.DataFrame(new_data, index=[0])
    
    result =  pipe.predict(new_data_df)

    return {"prediction": result}

@app.get("/health")
def health():
    return {"status": "healthy"}
