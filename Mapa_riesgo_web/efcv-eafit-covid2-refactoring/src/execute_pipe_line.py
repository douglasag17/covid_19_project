import json
from clusters import *
from model import *
from generate_data_set import generate_raw_embeddings
import pandas as pd
import json
import datetime as dt
from lambda_function import generate_dataset
import boto3

#refactored neeeded : Enviroment variables? config file?

NUM_CLUSTERS = 6
UPDATE_DATA_SET = True

if UPDATE_DATA_SET:
    generate_dataset()



data_embeddings = generate_raw_embeddings()
normalized_df = (data_embeddings-data_embeddings.mean())/data_embeddings.std()
normalized_df = normalized_df.fillna(0)

clusters = autoencoders_kmeans(normalized_df)
original_dimension = len(normalized_df.columns)
clusters.reduce_dimensionality(original_dimension)
clusters.fit_clusters(NUM_CLUSTERS)
timestamp = dt.datetime.utcnow().strftime("%Y%m%d")

with open('nombreDepartamento_codigo.json') as f:
    keys_for_plot = json.load(f)

cast_dic = {}
aux_file = clusters.clusters
for keys in aux_file.keys():
    codigo = keys_for_plot[keys]
    cast_dic[codigo] = int(aux_file[keys])
print(timestamp)
s3 = boto3.resource('s3')
obj = s3.Object('factored-eafit-bucket', 'semaforo/data_base/' + timestamp + '.json')
obj.put(Body=json.dumps(cast_dic))




