import json
import pandas as pd
from sodapy import Socrata
from datetime import datetime, timedelta, date
from unidecode import unidecode
import copy
from io import StringIO
from update_data import *

BUCKET_NAME = "factored-eafit-bucket"
INFECTED_DEMOGRAPHIC_FILE = 'sird_constantes_dia.csv'

def generate_dataset():
    # get data from csv files
    print('dowloading morbility data')
    df_morbilidades = get_morbidity("MORBILIDAD.csv")
    print('dowloading demographic data')
    df_densidad_poblacional = get_population_density(
        "densidad_poblacional_x_departamento.csv"
    )
    print('dowloading age data')
    df_edad_promedio = get_avg_age("edad_promedio.csv")
    print('dowloading economic data')
    df_indice_pobreza = get_poverty_index("indice_pobreza.csv")
    df_mayores_65 = read_csv_url("mayores_65.csv")

    print('preparing data set')
    # merging df_morbilidades, df_densidad_poblacional, df_edad_promedio, df_indice_pobreza, df_mayores_65
    df_variables_constantes = merge_dfs_const(
        df_morbilidades,
        df_densidad_poblacional,
        df_edad_promedio,
        df_indice_pobreza,
        df_mayores_65,
    )

    print('get constants per day')
    df_variables_constantes_dia = get_const_per_day(df_variables_constantes)

    print('get data of infected, recovered and deceased using the api of ins')
    df_cuenta_departamento_dia = get_ird()

    print('merging df_cuenta_departamento_dia with df_variables_constantes_dia')
    df_ird_constantes_dia = merge_ird_const(
        df_cuenta_departamento_dia, df_variables_constantes_dia
    )

    print('compute suceptibles')
    df_sird_constantes_dia = calculate_susceptible(df_ird_constantes_dia)

    print('get average morbidity and drop the other morbidity variables')
    df_sird_constantes_dia = get_avg_morbidity(df_sird_constantes_dia)
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y")
    print(timestampStr)
    # write csv
    csv_buffer = StringIO()
    df_sird_constantes_dia.to_csv(csv_buffer, index=False)

    
    s3_resource = boto3.resource('s3')
    s3_resource.Object(BUCKET_NAME, 'semaforo/' + INFECTED_DEMOGRAPHIC_FILE).put(Body=csv_buffer.getvalue())
    return "Succesfully written data base"

if __name__ == '__main__':
    generate_dataset()