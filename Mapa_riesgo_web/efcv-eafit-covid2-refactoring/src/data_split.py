import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

cross_validation_out_time = []
out_of_space = {}
data_base = pd.read_csv('data/23-Aug-2020sird_constantes_dia.csv')
regions = data_base.departamento.unique()
dates = data_base.fecha.unique()
time_stamp = data_base.fecha.unique()
splits = [ i*0.1 for i in range(4,11,2)]

for split in splits:
    dates_split = dates[:(int(split*len(dates)))]
    threshold = int(0.96 * len(dates_split))
    temp_data_train = data_base[data_base.fecha.isin(dates_split[:threshold])]
    temp_data_test = data_base[data_base.fecha.isin(dates_split[threshold:])]
    aux = {}
    aux['train'] = temp_data_train
    aux['test'] = temp_data_test
    cross_validation_out_time.append(aux)
    
train_regions = np.random.choice(regions,int(0.9 * len(regions)))
test_regions = set(regions) - set(train_regions)
out_of_space['train'] = data_base[data_base.departamento.isin(train_regions)]
out_of_space['test'] = data_base[data_base.departamento.isin(test_regions)]
with open('data/train_test_out_time.pkl','wb') as file:
    pkl.dump(cross_validation_out_time, file)
cross_validation_out_time[-1]['train'].to_csv('data/train.csv')
cross_validation_out_time[-1]['test'].to_csv('data/test.csv')
with open('data/train_test_out_space.pkl','wb') as file:
    pkl.dump(out_of_space, file)