import pandas as pd
import tensorflow as tf
import numpy as np
from model import get_model
from model import model
import datetime
import pickle as pkl

class Dataset_allSeries():
    def __init__(self, window, forecast_window, departamentos):
        self.smoothing_param = 0.1
        self.max_infectados = {}
        self.departamentos = departamentos
        self.window_size = window
        self.max_infectados_vector = np.array([])
        self.forecast_window = forecast_window

    def get_data_set(self, data, train_days, test_days):
        train_data = data['train']
        test_data = data['test']

        features_raw = pd.DataFrame()
        features_exp_smooth = pd.DataFrame()
        features_exp_smooth_test = pd.DataFrame()
        labels_raw = pd.DataFrame()
        features_test_raw = pd.DataFrame()
        labels_test_raw = pd.DataFrame()
        print("Cantidad de departamentos", len(self.departamentos))
        for departamento in self.departamentos:
            features_data = train_data[train_data.departamento == departamento].reset_index()

            labels_data = train_data[train_data.departamento == departamento][self.window_size:].reset_index()

            features_test = train_data[train_data.departamento == departamento][-self.window_size:].reset_index()
            labels_test = test_data[test_data.departamento == departamento].reset_index()

            self.max_infectados[departamento] = max(features_data.infectados)
            self.max_infectados_vector = np.append(self.max_infectados_vector,self.max_infectados[departamento])
            
            aux = smooth_infected(self.smoothing_param, features_data.infectados.values)
            # print(departamento)
            features_raw[departamento] = (np.log(features_data.infectados + np.finfo(float).eps) )
            features_exp_smooth[departamento + 'smoothed'] = aux
            labels_raw[departamento] = labels_data.infectados

            features_test_raw[departamento] = (np.log(features_test.infectados + np.finfo(float).eps))
            features_exp_smooth_test[departamento + 'smoothed'] = aux[-self.window_size:]
            labels_test_raw[departamento] =  labels_test.infectados
                
        data = tf.data.Dataset.from_tensor_slices(features_raw.values)
        data_labels = tf.data.Dataset.from_tensor_slices(labels_raw.values)
        data_test = tf.data.Dataset.from_tensor_slices(features_test_raw.values)
        data_labels_test = tf.data.Dataset.from_tensor_slices(labels_test_raw.values)

        days_train = tf.data.Dataset.from_tensor_slices(train_days)
        days_train_pred = tf.data.Dataset.from_tensor_slices(train_days[self.window_size:])
        days_test = tf.data.Dataset.from_tensor_slices(train_days[-self.window_size:] + test_days)
        days_test_pred =  tf.data.Dataset.from_tensor_slices(test_days)

        smooth = tf.data.Dataset.from_tensor_slices(features_exp_smooth.values)
        smooth_test = tf.data.Dataset.from_tensor_slices(features_exp_smooth_test.values)


        features = data.window(self.window_size, shift=1, stride=1, drop_remainder=True)
        labels = data_labels.window(self.forecast_window, shift=1, stride=1, drop_remainder=True)
        features_test = data_test.window(self.window_size, shift=1, stride=1, drop_remainder=True)
        labels_test = data_labels_test.window(self.forecast_window, shift=1, stride=1, drop_remainder=True)

        features_smooth = smooth.window(self.window_size, shift=1, stride=1, drop_remainder=True)
        features_smooth_test = smooth_test.window(self.window_size, shift=1, stride=1, drop_remainder=True)

        days_training_1 = days_train.window(self.window_size, shift=1, stride=1, drop_remainder=True)
        days_training_2 = days_train_pred.window(self.forecast_window, shift=1, stride=1, drop_remainder=True)
        days_test_1 = days_test.window(self.window_size, shift=1, stride=1, drop_remainder=True)
        days_test_2 = days_test_pred.window(self.forecast_window, shift=1, stride=1, drop_remainder=True)

        features = features.flat_map(lambda x:x.batch(self.window_size))
        labels = labels.flat_map(lambda x:x.batch(self.forecast_window))
        features_test = features_test.flat_map(lambda x:x.batch(self.window_size))
        labels_test = labels_test.flat_map(lambda x:x.batch(self.forecast_window))

        features_smooth = features_smooth.flat_map(lambda x:x.batch(self.window_size))
        features_smooth_test = features_smooth_test.flat_map(lambda x:x.batch(self.window_size))

        days_training_1 = days_training_1.flat_map(lambda x:x.batch(self.window_size))
        days_training_2 = days_training_2.flat_map(lambda x:x.batch(self.forecast_window))
        days_test_1 = days_test_1.flat_map(lambda x:x.batch(self.window_size))
        days_test_2 = days_test_2.flat_map(lambda x:x.batch(self.forecast_window))


        dataset = tf.data.Dataset.zip((features,features_smooth,days_training_1,days_training_2))
        val_dataset = tf.data.Dataset.zip((features_test,features_smooth_test,days_test_1, days_test_2)) 

        dataset = tf.data.Dataset.zip((dataset,labels))
        val_dataset = tf.data.Dataset.zip((val_dataset,labels_test))

        return dataset, val_dataset

def smooth_infected(alpha, vector_infected):
    aux = []
    smoothed_infected = 0
    for i in range(len(vector_infected)):
        smoothed_infected = alpha*vector_infected[i] +\
                                (1-alpha)*smoothed_infected
                                
        aux.append(smoothed_infected)
    return aux

def generate_seq(until, lenght, starts_with=0):
    aux = []
    for i in range(starts_with, starts_with + lenght):
        aux.append(i % until)
    return aux
    
def mae_imp(y_true, y_pred):
    return tf.reduce_sum(tf.math.reduce_mean(tf.math.abs(tf.math.subtract(y_true,y_pred)), axis=1),axis = -1)

# def mae_imp(y_true, y_pred):
#     return tf.reduce_sum(tf.math.reduce_mean(tf.math.abs(tf.math.subtract(y_true,y_pred)), axis=1),axis = -1)

if __name__ == "__main__":
    data_aux = pd.read_csv('../data/sird_constantes_dia.csv')
    fechas = data_aux.fecha.unique()
    init_date = fechas[len(fechas)-7]
    end_date = fechas[-1]
    print(init_date, len(fechas)-7, end_date)
    train_set = data_aux[data_aux.fecha.isin(fechas[:len(fechas)-7])]
    test_set =  data_aux[data_aux.fecha.isin(fechas[len(fechas)-7:len(fechas)-7 + 7])]
    # Biggest subset
    data = {'train':train_set, 'test': test_set}
    departamentos = data['train']["departamento"].unique()
    #departamentos = ['atlantico' , 'bogota', 'valle del cauca', 'antioquia']
    window = 10
    forecast_window = 7
    num_dep = len(departamentos)
    train_days = generate_seq(7, 164)
    test_days = generate_seq(7, 9, (train_days[-1] + 1)%7)

    dataset = Dataset_allSeries(window, forecast_window, departamentos)
    train_dataset, val_dataset = dataset.get_data_set(data, train_days, test_days)

    net = get_model('model.h5', forecast_window, window, num_dep)
    
    net.compile(loss=mae_imp, optimizer="adam")
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', #val_loss
                                                   patience=8,
                                                   mode='min',
                                                   restore_best_weights=True)
    history = net.fit(train_dataset.batch(32), 
                    validation_data=val_dataset.batch(1),
                    epochs=5, 
                    callbacks = [es_callback])
    print(val_dataset)
    preds = net.predict(val_dataset.batch(1))
    with open('preds_{}_{}.npy'.format(init_date,end_date), 'wb') as f:
        np.save(f, preds)