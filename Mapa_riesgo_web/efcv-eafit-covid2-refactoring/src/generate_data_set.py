import pandas as pd
import boto3
from update_data import read_csv_url
from feature_extraction import smooth_infected
import tensorflow as tf
import numpy as np
from model import get_model
from feature_extraction import Dataset_allSeries



BUCKET_NAME = "factored-eafit-bucket"
INFECTED_DEMOGRAPHIC_FILE = 'sird_constantes_dia.csv'
ECONOMIC_CLIMATE = 'economicas_clima.csv'
#include deaths and recovery predictions
RELEVANT_INDEXES = ['fecha', 'departamento', 'infectados',
                    'cantidad_mayores_65', 'ipm', 'poblacion_total',
                    'personas_km2', 'edad_promedio', 'promedio_morbilidades']
window_model=10
WINDOW_TRAINING = 10
WINDOW_FORECAST = 7

def generate_raw_embeddings():
    data_set = read_csv_url(ECONOMIC_CLIMATE)
    data_set = data_set.set_index('Unnamed: 0', drop=True)
    data_set = data_set.drop(columns = "('Unnamed: 1_level_0', 'DPTO_CCDGO')")
    indexes = data_set.index
    tabla = read_csv_url(INFECTED_DEMOGRAPHIC_FILE)
    data = tabla[RELEVANT_INDEXES]
    last_date = data.fecha.iloc[-1]
    fechas = tabla.fecha.unique()
    test_fechas = fechas[-7:]
    data_split = {'train':data[~data.fecha.isin(test_fechas)], 'test': data[data.fecha.isin(test_fechas)]}
    departamentos = data.departamento.unique()
    #departamentos = ['atlantico' , 'bogota', 'valle del cauca', 'antioquia']
    window = 10
    forecast_window = 7
    num_dep = len(departamentos)
    num_days = len(data.fecha.unique())
    def generate_seq(until, lenght, starts_with=0):
        aux = []
        for i in range(starts_with, starts_with + lenght):
            aux.append(i % until)
        return aux
    train_days = generate_seq(7, num_days)
    test_days = generate_seq(7, 9, (train_days[-1] + 1)%7)
    dataset = Dataset_allSeries(window, forecast_window, departamentos)
    train_dataset, val_dataset = dataset.get_data_set(data_split, train_days, test_days)
    def mae_imp(y_true, y_pred):
        return tf.reduce_sum(tf.math.reduce_mean(tf.math.abs(tf.math.subtract(y_true,y_pred)), axis=1),axis = -1)
    net = get_model('model.h5', WINDOW_FORECAST, WINDOW_TRAINING, num_dep)
    net.compile(loss=mae_imp,
    optimizer="adam")
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', #val_loss
                                            patience=8,
                                            mode='min',
                                                restore_best_weights=True)
    history = net.fit(train_dataset.batch(32), 
                    validation_data=val_dataset.batch(1),
                    epochs=150, 
                    callbacks = [es_callback])
    net.save_weights('model.h5')
    
    preds = net.predict(val_dataset.batch(1))
    predicted_infected = np.mean(preds, axis=1)

    data_embeddings = data[data.fecha == last_date]
    predicted_infected = tf.squeeze(predicted_infected)
    data_embeddings['infectados'].drop(columns='infectados', inplace=True)
    data_embeddings['infectados'] = tf.make_ndarray(tf.make_tensor_proto(predicted_infected))
    data_embeddings.set_index('departamento', drop=True, inplace=True)
    data_embeddings = data_embeddings.drop(columns=['fecha'])
    data_embeddings = data_embeddings.sort_index()


    aux_index = data_embeddings.index
    data_set = data_set.set_index(aux_index)
    data_embeddings = data_embeddings.join(data_set)

    return data_embeddings
