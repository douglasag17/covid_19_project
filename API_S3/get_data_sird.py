from fastapi import FastAPI
import pandas as pd
import boto3
import json
import io
from datetime import datetime, timedelta


app = FastAPI()

AWS_ACCESS_KEY_ID = 'AWS_ACCESS_KEY_ID'
AWS_SECRET_ACCESS_KEY = 'AWS_SECRET_ACCESS_KEY'
BUCKET = 'factored-eafit-bucket'
FILE = 'semaforo/sird_constantes_dia.csv'
FILE_2 = 'semaforo/data_base/'

@app.get("/data_sird")
async def root():

    s3_resource = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    csv_obj = s3_resource.get_object(Bucket=BUCKET, Key=FILE)
    body = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(body))
    result = df.to_json(orient="records")
    parsed = json.loads(result)
    return json.dumps(parsed)

@app.get("/get_clusters")
async def root():
    timestamp = (datetime.utcnow() - timedelta(days=1)).strftime("%d%m%Y")
    s3_resource = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    content_object = s3_resource.get_object(Bucket=BUCKET, Key=f"{FILE_2}{timestamp}.json")
    file_content = content_object['Body'].read().decode('utf-8')
    departments = json.loads(file_content)
    return departments


# uvicorn get_data_sird:app --reload