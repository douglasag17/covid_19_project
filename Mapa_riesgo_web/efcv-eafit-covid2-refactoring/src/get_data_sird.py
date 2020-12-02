import pandas as pd
import boto3
import json
import io
from fastapi import FastAPI



app = FastAPI()

AWS_ACCESS_KEY_ID = 'AWS_ACCESS_KEY_ID'
AWS_SECRET_ACCESS_KEY = 'AWS_SECRET_ACCESS_KEY'
BUCKET = 'factored-eafit-bucket'
FILE = 'semaforo/sird_constantes_dia.csv'


@app.get("/data_sird")
async def root():

    s3_resource = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    csv_obj = s3_resource.get_object(Bucket=BUCKET, Key=FILE)
    body = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(body))
    result = df.to_json(orient="records")
    parsed = json.loads(result)
    return json.dumps(parsed)

# uvicorn get_data_sird:app --reload