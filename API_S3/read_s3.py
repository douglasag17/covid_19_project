import boto3
import json
import io
import pandas as pd

# TODO variables globales

s3_resource = boto3.client('s3', aws_access_key_id='AKIATUDN23RNKTQVVLN7', aws_secret_access_key='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')
csv_obj = s3_resource.get_object(Bucket='factored-eafit-bucket', Key='semaforo/' + 'densidad_poblacional_x_departamento.csv')
body = csv_obj['Body'].read().decode('utf-8')
df = pd.read_csv(io.StringIO(body))

result = df.to_json(orient="records")
parsed = json.loads(result)

print(json.dumps(parsed, indent=4) )
