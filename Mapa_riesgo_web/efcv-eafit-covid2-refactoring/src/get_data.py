from fastapi import FastAPI
import datetime
import pandas as pd
import boto3
import json
import io


app = FastAPI()

BUCKET = 'factored-eafit-bucket'
FILE = 'semaforo/data_base/'



@app.get("/get_clusters")
async def root():
    timestamp = (dt.datetime.utcnow() - timedelta(days=1)).strftime("%d%m%Y")
    s3_resource = boto3.client('s3')
    content_object = s3_resource.get_object(Bucket=BUCKET, Key=FILE + timestamp + '.json')
    file_content = content_object.get()['Body'].read().decode('utf-8')
    departments = json.loads(file_content)
    print(departments)
    return departments

# uvicorn main:app --reload