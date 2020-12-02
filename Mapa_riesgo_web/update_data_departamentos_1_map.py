# Esto se ejecuta cada semana (todos los lunes)

import pandas as pd
from sodapy import Socrata
from datetime import datetime, timedelta, date
from unidecode import unidecode
import copy
import numpy as np
import predict_incidencias


def read_csv_url(url):
    df = pd.read_csv(url)
    return df


def get_population_density(source_data):

    # read data from csv
    df_densidad_poblacional = read_csv_url(source_data)

    # transformations
    df_densidad_poblacional["departamento"] = df_densidad_poblacional[
        "departamento"
    ].str.lower()
    df_densidad_poblacional["departamento"] = df_densidad_poblacional[
        "departamento"
    ].transform(lambda x: unidecode(x))
    df_densidad_poblacional["departamento"] = df_densidad_poblacional[
        "departamento"
    ].transform(lambda x: x if x != "bogota d.c" else "bogota")
    df_densidad_poblacional = df_densidad_poblacional[
        df_densidad_poblacional["departamento"] != "colombia"
    ]
    df_densidad_poblacional = df_densidad_poblacional.drop(
        columns=["superficie(km2)", "personas por km2"]
    )

    return df_densidad_poblacional


def get_const_per_day(df_variables_constantes):

    # Dataframe variables constantes por dia
    df_variables_constantes_dia = pd.DataFrame(
        columns=["fecha"] + df_variables_constantes.columns.to_list()
    )
    df_fecha = pd.date_range(
        start=datetime(2020, 3, 2), end=datetime.now() - timedelta(days=2)
    ).to_frame(index=False, name="fecha")
    for i, row_i in df_fecha.iterrows():
        for j, row_j in df_variables_constantes.iterrows():
            values = {
                "fecha": datetime.strftime(row_i.fecha, "%Y-%m-%d"),
                "departamento": row_j.departamento,
                "poblacion": row_j.poblacion,
            }
            df_variables_constantes_dia = df_variables_constantes_dia.append(
                values, ignore_index=True
            )
    return df_variables_constantes_dia


def merge_ird_const(df_cuenta_departamento_dia, df_variables_constantes_dia):

    # Total x dia x departamento + variables constantes
    df_ird_constantes_dia = df_cuenta_departamento_dia.merge(
        df_variables_constantes_dia,
        how="outer",
        left_on=["fecha", "departamento"],
        right_on=["fecha", "departamento"],
    )
    df_ird_constantes_dia = df_ird_constantes_dia[
        ["fecha", "departamento", "infectados",
            "recuperados", "decesos", "poblacion"]
    ]
    df_ird_constantes_dia = df_ird_constantes_dia[
        df_ird_constantes_dia["fecha"].notna()
    ]
    df_ird_constantes_dia = df_ird_constantes_dia.fillna(0)
    df_ird_constantes_dia = df_ird_constantes_dia.sort_values(
        by=["fecha", "departamento"]
    ).reset_index(drop=True)

    return df_ird_constantes_dia


def get_ird():

    # get to the API
    client = Socrata("www.datos.gov.co", "6aek14sky6N2pVL12sw1qfzoQ")
    # fecha_de_notificaci_n, fecha_inicio_sintomas, fecha_diagnostico
    select = "id_de_caso, departamento_nom, fecha_muerte, fecha_recuperado, fecha_reporte_web"
    results = client.get("gt2j-8ykr", limit=5000000, select=select)
    df_casos = pd.DataFrame.from_records(results)
    df_casos = df_casos.fillna("-   -")
    df_casos = df_casos.rename(columns={"departamento_nom": "departamento"})

    # Transformacion campos tipo fecha
    df_casos["fecha_muerte"] = df_casos["fecha_muerte"].transform(
        lambda x: x[:10].split(" ")[0] if x != "-   -" else x
    )
    df_casos["fecha_recuperado"] = df_casos["fecha_recuperado"].transform(
        lambda x: x[:10].split(" ")[0] if x != "-   -" else x
    )
    df_casos["fecha_reporte_web"] = df_casos["fecha_reporte_web"].transform(
        lambda x: x[:10].split(" ")[0] if x != "-   -" else x
    )

    # Formato de fechas
    df_casos["fecha_muerte"] = df_casos["fecha_muerte"].apply(
        lambda x: x
        if x == "-   -"
        else datetime.strptime(x, "%d/%m/%Y").strftime("%Y-%m-%d")
    )
    df_casos["fecha_recuperado"] = df_casos["fecha_recuperado"].apply(
        lambda x: x
        if x == "-   -"
        else datetime.strptime(x, "%d/%m/%Y").strftime("%Y-%m-%d")
    )
    df_casos["fecha_reporte_web"] = df_casos["fecha_reporte_web"].apply(
        lambda x: x
        if x == "-   -"
        else datetime.strptime(x, "%d/%m/%Y").strftime("%Y-%m-%d")
    )

    # Tranformacion nombres departamentos
    df_casos["departamento"] = df_casos["departamento"].str.lower()
    df_casos["departamento"] = df_casos["departamento"].transform(
        lambda x: unidecode(x)
    )
    df_casos["departamento"] = df_casos["departamento"].transform(
        lambda x: x if x != "bogota d.c." else "bogota"
    )
    df_casos["departamento"] = df_casos["departamento"].transform(
        lambda x: x if "archipielago" not in x else "san andres y providencia"
    )
    df_casos["departamento"] = df_casos["departamento"].transform(
        lambda x: x if x != "cartagena d.t. y c." else "bolivar"
    )
    df_casos["departamento"] = df_casos["departamento"].transform(
        lambda x: x if x != "barranquilla d.e." else "atlantico"
    )
    df_casos["departamento"] = df_casos["departamento"].transform(
        lambda x: x if x != "buenaventura d.e." else "valle del cauca"
    )
    df_casos["departamento"] = df_casos["departamento"].transform(
        lambda x: x if x != "santa marta d.t. y c." else "magdalena"
    )

    # Total x dia x departamento
    df_cuenta_departamento_dia = (
        df_casos.groupby(["fecha_reporte_web", "departamento"])["id_de_caso"]
        .count()
        .reset_index()
        .rename(columns={"id_de_caso": "infectados"})
    )
    df_fallecidos_departamento_dia = (
        df_casos[df_casos["fecha_muerte"] != "-   -"]
        .groupby(["fecha_reporte_web", "departamento"])["fecha_muerte"]
        .count()
        .reset_index()
        .rename(columns={"fecha_muerte": "decesos"})
    )
    df_recuperados_departamento_dia = (
        df_casos[df_casos["fecha_recuperado"] != "-   -"]
        .groupby(["fecha_reporte_web", "departamento"])["fecha_recuperado"]
        .count()
        .reset_index()
        .rename(columns={"fecha_recuperado": "recuperados"})
    )

    df_cuenta_departamento_dia = df_cuenta_departamento_dia.merge(
        df_fallecidos_departamento_dia,
        how="outer",
        left_on=["fecha_reporte_web", "departamento"],
        right_on=["fecha_reporte_web", "departamento"],
    )
    df_cuenta_departamento_dia = df_cuenta_departamento_dia.merge(
        df_recuperados_departamento_dia,
        how="outer",
        left_on=["fecha_reporte_web", "departamento"],
        right_on=["fecha_reporte_web", "departamento"],
    )
    df_cuenta_departamento_dia = df_cuenta_departamento_dia.fillna(0)

    # rename fecha
    df_cuenta_departamento_dia = df_cuenta_departamento_dia.rename(
        columns={"fecha_reporte_web": "fecha"}
    )

    return df_cuenta_departamento_dia


def calculate_susceptible(df_ird_constantes_dia):
    df_ird_constantes_dia = df_ird_constantes_dia[
        df_ird_constantes_dia["fecha"] != "-   -"
    ]

    df_sird_constantes_dia = copy.deepcopy(df_ird_constantes_dia)
    df_sird_constantes_dia["susceptibles"] = df_sird_constantes_dia[
        df_sird_constantes_dia["fecha"] == "2020-03-02"
    ]["poblacion"]

    df_sird_constantes_dia = df_sird_constantes_dia.sort_values(
        by=["departamento", "fecha"]
    ).reset_index(drop=True)
    for i, row in df_sird_constantes_dia.iterrows():
        if row["fecha"] != "2020-03-02":
            df_sird_constantes_dia.loc[i, "susceptibles"] = (
                df_sird_constantes_dia.loc[i - 1, "susceptibles"]
                - df_sird_constantes_dia.loc[i - 1, "infectados"]
                - df_sird_constantes_dia.loc[i - 1, "recuperados"]
                - df_sird_constantes_dia.loc[i - 1, "decesos"]
            )
    df_sird_constantes_dia = df_sird_constantes_dia.sort_values(
        by=["fecha", "departamento"]
    ).reset_index(drop=True)

    return df_sird_constantes_dia


def main():

    # # get poblacion total
    # df_densidad_poblacional = get_population_density(
    #     "./data/densidad_poblacional_x_departamento.csv"
    # )

    # # get constants per day
    # df_densidad_poblacional = get_const_per_day(df_densidad_poblacional)

    # # get data of infected, recovered and deceased using the api of ins
    # df_cuenta_departamento_dia = get_ird()

    # # merge df_cuenta_departamento_dia with df_densidad_poblacional
    # df_cuenta_departamento_dia = merge_ird_const(
    #     df_cuenta_departamento_dia, df_densidad_poblacional
    # )

    # # calculate susceptible
    # df_sird_constantes_dia = calculate_susceptible(df_cuenta_departamento_dia)
    # list_dep = ['amazonas',
    #             'antioquia',
    #             'arauca',
    #             'atlantico',
    #             'bogota',
    #             'bolivar',
    #             'boyaca',
    #             'caldas',
    #             'caqueta',
    #             'casanare',
    #             'cauca',
    #             'cesar',
    #             'choco',
    #             'cordoba',
    #             'cundinamarca',
    #             'guainia',
    #             'guaviare',
    #             'huila',
    #             'la guajira',
    #             'magdalena',
    #             'meta',
    #             'narino',
    #             'norte de santander',
    #             'putumayo',
    #             'quindio',
    #             'risaralda',
    #             'san andres y providencia',
    #             'santander',
    #             'sucre',
    #             'tolima',
    #             'valle del cauca',
    #             'vaupes',
    #             'vichada']
    # df_sird_constantes_dia = df_sird_constantes_dia[df_sird_constantes_dia['departamento'].isin(
    #     list_dep)]
    # df_sird_constantes_dia.to_csv(
    #     "./data/sird_constantes_dia.csv", index=False)

    # # get average SIRD
    # df_sird_prom = df_sird_constantes_dia.groupby(
    #     ["departamento"])["susceptibles", "infectados", "recuperados", "decesos"].mean()

    # # read constants variables
    # data_embeddings = pd.read_csv("./data/variables_constantes.csv")
    # data_embeddings = data_embeddings.drop(
    #     ["susceptibles", "infectados", "recuperados", "decesos"], 1
    # )

    # # merge data_embeddings with df_sird_prom
    # data_embeddings = pd.merge(
    #     data_embeddings, df_sird_prom, on="departamento")

    # get incidents predicted for the following week
    predict_incidencias.main()
    exit(1)

    # merge data_embeddings with incidents predicted
    preds = np.load("./data/preds_incidencias.npy")
    mean_preds = np.apply_along_axis(np.mean, 1, preds).transpose()
    incid = pd.DataFrame(mean_preds)
    departamentos = pd.DataFrame(data_embeddings["departamento"])
    departamentos = departamentos.sort_values(by=["departamento"])
    departamentos["inc_pred"] = incid
    data_embeddings = pd.merge(
        data_embeddings, departamentos, on="departamento")

    # write csv
    data_embeddings.to_csv("./data/sird_dia_embeddings.csv", index=False)


main()
