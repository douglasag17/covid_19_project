import pandas as pd
from sodapy import Socrata
from datetime import datetime, timedelta, date
from unidecode import unidecode
import copy


def read_csv_url(url):
    df = pd.read_csv(url)
    return df


def get_morbidity(source_data):

    # read data from csv
    df_morbilidades_crudo = read_csv_url(source_data)

    # transformations
    df_morbilidades_crudo = df_morbilidades_crudo.rename(
        columns={
            "Gran causa de morbilidad": "causa_morbilidad",
            "Subgrupo de causas": "morbilidad",
            "Sexo": "sexo",
            "Núm.  Personas 2012": "cantidad_morbilidad",
        }
    )
    df_morbilidades_crudo = df_morbilidades_crudo.drop(
        columns=["id_depto", "causa_morbilidad", "sexo"]
    )
    df_morbilidades_crudo["nom_depto"] = df_morbilidades_crudo["nom_depto"].str.lower()
    df_morbilidades_crudo["nom_depto"] = df_morbilidades_crudo["nom_depto"].transform(
        lambda x: unidecode(x)
    )
    df_morbilidades_crudo["nom_depto"] = df_morbilidades_crudo["nom_depto"].transform(
        lambda x: x
        if x != "archipielago de san andres, providencia y santa catalina"
        else "san andres y providencia"
    )
    df_morbilidades_crudo["nom_depto"] = df_morbilidades_crudo["nom_depto"].transform(
        lambda x: x if x != "bogota d.c" else "bogota"
    )
    df_morbilidades_crudo = df_morbilidades_crudo.dropna(subset=["morbilidad"])
    df_morbilidades_crudo = df_morbilidades_crudo.sort_values(
        by=["nom_depto", "morbilidad"]
    ).reset_index(drop=True)
    df_morbilidades_crudo["morbilidad"] = df_morbilidades_crudo[
        "morbilidad"
    ].str.lower()
    df_morbilidades_crudo["morbilidad"] = df_morbilidades_crudo["morbilidad"].transform(
        lambda x: unidecode(x)
    )
    df_morbilidades_crudo["morbilidad"] = df_morbilidades_crudo["morbilidad"].transform(
        lambda x: x.replace(" ", "_")
    )
    df_morbilidades_crudo["morbilidad"] = df_morbilidades_crudo["morbilidad"].transform(
        lambda x: x.replace(",", "_")
    )
    df_morbilidades_crudo["morbilidad"] = df_morbilidades_crudo["morbilidad"].transform(
        lambda x: x.replace("-", "_")
    )

    # agrupar por morbilidad sumando cantidad_morbilidad
    df_morbilidades_departamento = (
        df_morbilidades_crudo.groupby(["morbilidad", "nom_depto"])[
            "cantidad_morbilidad"
        ]
        .sum()
        .reset_index()
    )
    df_morbilidades_departamento = df_morbilidades_departamento.sort_values(
        by=["nom_depto", "morbilidad"]
    ).reset_index(drop=True)

    # morbilidades queden como columnas
    df_morbilidades = df_morbilidades_departamento.pivot(
        index="nom_depto", columns="morbilidad", values="cantidad_morbilidad"
    )
    df_morbilidades["departamento"] = df_morbilidades.index
    df_morbilidades = df_morbilidades.reset_index(drop=True)

    return df_morbilidades


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
    df_densidad_poblacional = df_densidad_poblacional.drop(columns=["superficie(km2)"])

    return df_densidad_poblacional


def get_avg_age(source_data):

    # read data from csv
    df_edad_promedio = read_csv_url(source_data)

    # transformations
    df_edad_promedio["departamento"] = df_edad_promedio["departamento"].transform(
        lambda x: unidecode(x)
    )

    return df_edad_promedio


def get_poverty_index(source_data):

    # read data from csv
    df_indice_pobreza_municipio = read_csv_url(source_data)

    # transformations
    df_indice_pobreza = (
        df_indice_pobreza_municipio.groupby("Codigo Departamento")["Total "]
        .mean()
        .reset_index()
        .rename(
            columns={"Codigo Departamento": "codigo_departamento", "Total ": "total"}
        )
    )

    return df_indice_pobreza


def merge_dfs_const(
    df_morbilidades,
    df_densidad_poblacional,
    df_edad_promedio,
    df_indice_pobreza,
    df_mayores_65,
):

    # Merge df_indice_pobreza con df_mayores_65
    df_pobreza_65 = df_indice_pobreza.merge(
        df_mayores_65, how="outer", left_on="codigo_departamento", right_on="Código",
    )
    df_pobreza_65 = df_pobreza_65[["Nombre del Departamento", "Total", "total"]]
    df_pobreza_65 = df_pobreza_65.rename(
        columns={
            "Nombre del Departamento": "departamento",
            "Total": "cantidad_mayores_65",
            "total": "ipm",
        }
    )
    df_pobreza_65["departamento"] = df_pobreza_65["departamento"].str.lower()
    df_pobreza_65["departamento"] = df_pobreza_65["departamento"].transform(
        lambda x: unidecode(x)
    )
    df_pobreza_65["departamento"] = df_pobreza_65["departamento"].transform(
        lambda x: unidecode(x)
        if x != "san andres, providencia y santa catalina"
        else "san andres y providencia"
    )
    df_pobreza_65 = df_pobreza_65.sort_values(by=["departamento"])

    # Merge df_densidad_poblacional con df_edad_promedio
    df_densidad_edad = df_densidad_poblacional.merge(
        df_edad_promedio, how="outer", left_on="departamento", right_on="departamento"
    )
    df_densidad_edad = df_densidad_edad.sort_values(by=["departamento"])
    df_densidad_edad = df_densidad_edad[
        ["departamento", "poblacion", "personas por km2", "edad_promedio",]
    ]
    df_densidad_edad = df_densidad_edad.rename(
        columns={"poblacion": "poblacion_total", "personas por km2": "personas_km2",}
    )
    df_densidad_edad = df_densidad_edad.drop([33])

    # Merge df_pobreza_65 con df_densidad_edad
    df_variables_constantes_1 = df_pobreza_65.merge(
        df_densidad_edad, how="outer", left_on="departamento", right_on="departamento"
    )

    # Merge variables constantes con morbilidades
    df_variables_constantes = df_variables_constantes_1.merge(
        df_morbilidades, how="inner", left_on="departamento", right_on="departamento"
    )

    return df_variables_constantes


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
                "cantidad_mayores_65": row_j.cantidad_mayores_65,
                "ipm": row_j.ipm,
                "poblacion_total": row_j.poblacion_total,
                "personas_km2": row_j.personas_km2,
                "edad_promedio": row_j.edad_promedio,
                "anomalias_congenitas": row_j.anomalias_congenitas,
                "codiciones_orales": row_j.codiciones_orales,
                "condiciones_derivadas_durante_el_periodo_perinatal": row_j.condiciones_derivadas_durante_el_periodo_perinatal,
                "condiciones_maternas": row_j.condiciones_maternas,
                "condiciones_neuropsiquiatricas": row_j.condiciones_neuropsiquiatricas,
                "deficiencias_nutricionales": row_j.deficiencias_nutricionales,
                "desordenes_endocrinos": row_j.desordenes_endocrinos,
                "diabetes_mellitus": row_j.diabetes_mellitus,
                "enfermedades_cardiovasculares": row_j.enfermedades_cardiovasculares,
                "enfermedades_de_la_piel": row_j.enfermedades_de_la_piel,
                "enfermedades_de_los_organos_de_los_sentidos": row_j.enfermedades_de_los_organos_de_los_sentidos,
                "enfermedades_digestivas": row_j.enfermedades_digestivas,
                "enfermedades_genitourinarias": row_j.enfermedades_genitourinarias,
                "enfermedades_infecciosas_y_parasitarias": row_j.enfermedades_infecciosas_y_parasitarias,
                "enfermedades_musculo_esqueleticas": row_j.enfermedades_musculo_esqueleticas,
                "enfermedades_respiratorias": row_j.enfermedades_respiratorias,
                "infecciones_respiratorias": row_j.infecciones_respiratorias,
                "lesiones_de_intensionalidad_indeterminada": row_j.lesiones_de_intensionalidad_indeterminada,
                "lesiones_intensionales": row_j.lesiones_intensionales,
                "lesiones_no_intensionales": row_j.lesiones_no_intensionales,
                "neoplasias_malignas": row_j.neoplasias_malignas,
                "otras_neoplasias": row_j.otras_neoplasias,
                "signos_y_sintomas_mal_definidos": row_j.signos_y_sintomas_mal_definidos,
                "traumatismos__envenenamientos_u_algunas_otras_consecuencias_de_causas_externas": row_j.traumatismos__envenenamientos_u_algunas_otras_consecuencias_de_causas_externas,
            }
            df_variables_constantes_dia = df_variables_constantes_dia.append(
                values, ignore_index=True
            )
    return df_variables_constantes_dia


def get_ird():

    # get to the API
    client = Socrata("www.datos.gov.co", "6aek14sky6N2pVL12sw1qfzoQ")
    select = "id_de_caso, fecha_de_notificaci_n, departamento, fis, fecha_de_muerte, fecha_diagnostico, fecha_recuperado, fecha_reporte_web"
    results = client.get("gt2j-8ykr", limit=5000000, select=select)
    df_casos = pd.DataFrame.from_records(results)
    df_casos = df_casos.fillna("-   -")

    # Transformacion campos tipo fecha
    df_casos["fecha_diagnostico"] = df_casos["fecha_diagnostico"].transform(
        lambda x: x[:10] if x != "-   -" else x
    )
    df_casos["fis"] = df_casos["fis"].transform(lambda x: x[:10] if x != "-   -" else x)
    df_casos["fecha_de_muerte"] = df_casos["fecha_de_muerte"].transform(
        lambda x: x[:10] if x != "-   -" else x
    )
    df_casos["fecha_de_notificaci_n"] = df_casos["fecha_de_notificaci_n"].transform(
        lambda x: x[:10] if x != "-   -" else x
    )
    df_casos["fecha_recuperado"] = df_casos["fecha_recuperado"].transform(
        lambda x: x[:10] if x != "-   -" else x
    )
    df_casos["fecha_reporte_web"] = df_casos["fecha_reporte_web"].transform(
        lambda x: x[:10] if x != "-   -" else x
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
        df_casos[df_casos["fecha_de_muerte"] != "-   -"]
        .groupby(["fecha_reporte_web", "departamento"])["fecha_de_muerte"]
        .count()
        .reset_index()
        .rename(columns={"fecha_de_muerte": "decesos"})
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

    return df_cuenta_departamento_dia


def merge_ird_const(df_cuenta_departamento_dia, df_variables_constantes_dia):

    # Total x dia x departamento + variables constantes
    df_ird_constantes_dia = df_cuenta_departamento_dia.merge(
        df_variables_constantes_dia,
        how="outer",
        left_on=["fecha_reporte_web", "departamento"],
        right_on=["fecha", "departamento"],
    )
    df_ird_constantes_dia = df_ird_constantes_dia[
        [
            "fecha",
            "departamento",
            "infectados",
            "recuperados",
            "decesos",
            "cantidad_mayores_65",
            "ipm",
            "poblacion_total",
            "personas_km2",
            "edad_promedio",
            "anomalias_congenitas",
            "codiciones_orales",
            "condiciones_derivadas_durante_el_periodo_perinatal",
            "condiciones_maternas",
            "condiciones_neuropsiquiatricas",
            "deficiencias_nutricionales",
            "desordenes_endocrinos",
            "diabetes_mellitus",
            "enfermedades_cardiovasculares",
            "enfermedades_de_la_piel",
            "enfermedades_de_los_organos_de_los_sentidos",
            "enfermedades_digestivas",
            "enfermedades_genitourinarias",
            "enfermedades_infecciosas_y_parasitarias",
            "enfermedades_musculo_esqueleticas",
            "enfermedades_respiratorias",
            "infecciones_respiratorias",
            "lesiones_de_intensionalidad_indeterminada",
            "lesiones_intensionales",
            "lesiones_no_intensionales",
            "neoplasias_malignas",
            "otras_neoplasias",
            "signos_y_sintomas_mal_definidos",
            "traumatismos__envenenamientos_u_algunas_otras_consecuencias_de_causas_externas",
        ]
    ]
    df_ird_constantes_dia = df_ird_constantes_dia[
        df_ird_constantes_dia["fecha"].notna()
    ]
    df_ird_constantes_dia = df_ird_constantes_dia.fillna(0)
    df_ird_constantes_dia = df_ird_constantes_dia.sort_values(
        by=["fecha", "departamento"]
    ).reset_index(drop=True)

    return df_ird_constantes_dia


def calculate_susceptible(df_ird_constantes_dia):
    df_sird_constantes_dia = copy.deepcopy(df_ird_constantes_dia)
    df_sird_constantes_dia["susceptibles"] = df_sird_constantes_dia[
        df_sird_constantes_dia["fecha"] == "2020-03-02"
    ]["poblacion_total"]

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


def get_avg_morbidity(df_sird_constantes_dia):

    # promedio morbilidades
    df_sird_constantes_dia["promedio_morbilidades"] = (
        df_sird_constantes_dia["anomalias_congenitas"]
        + df_sird_constantes_dia["codiciones_orales"]
        + df_sird_constantes_dia["condiciones_derivadas_durante_el_periodo_perinatal"]
        + df_sird_constantes_dia["condiciones_maternas"]
        + df_sird_constantes_dia["condiciones_neuropsiquiatricas"]
        + df_sird_constantes_dia["deficiencias_nutricionales"]
        + df_sird_constantes_dia["desordenes_endocrinos"]
        + df_sird_constantes_dia["diabetes_mellitus"]
        + df_sird_constantes_dia["enfermedades_cardiovasculares"]
        + df_sird_constantes_dia["enfermedades_de_la_piel"]
        + df_sird_constantes_dia["enfermedades_de_los_organos_de_los_sentidos"]
        + df_sird_constantes_dia["enfermedades_digestivas"]
        + df_sird_constantes_dia["enfermedades_genitourinarias"]
        + df_sird_constantes_dia["enfermedades_infecciosas_y_parasitarias"]
        + df_sird_constantes_dia["enfermedades_musculo_esqueleticas"]
        + df_sird_constantes_dia["enfermedades_respiratorias"]
        + df_sird_constantes_dia["infecciones_respiratorias"]
        + df_sird_constantes_dia["lesiones_de_intensionalidad_indeterminada"]
        + df_sird_constantes_dia["lesiones_intensionales"]
        + df_sird_constantes_dia["lesiones_no_intensionales"]
        + df_sird_constantes_dia["neoplasias_malignas"]
        + df_sird_constantes_dia["otras_neoplasias"]
        + df_sird_constantes_dia["signos_y_sintomas_mal_definidos"]
        + df_sird_constantes_dia[
            "traumatismos__envenenamientos_u_algunas_otras_consecuencias_de_causas_externas"
        ]
    ) / 24

    df_sird_constantes_dia = df_sird_constantes_dia[
        [
            "fecha",
            "departamento",
            "susceptibles",
            "infectados",
            "recuperados",
            "decesos",
            "cantidad_mayores_65",
            "ipm",
            "poblacion_total",
            "personas_km2",
            "edad_promedio",
            "promedio_morbilidades",
        ]
    ]
    df_sird_constantes_dia = df_sird_constantes_dia.sort_values(
        by=["fecha", "departamento"]
    ).reset_index(drop=True)
    return df_sird_constantes_dia


def main():

    # get data from csv files
    df_morbilidades = get_morbidity("./data/MORBILIDAD.csv")
    df_densidad_poblacional = get_population_density(
        "./data/densidad_poblacional_x_departamento.csv"
    )
    df_edad_promedio = get_avg_age("./data/edad_promedio.csv")
    df_indice_pobreza = get_poverty_index("./data/indice_pobreza.csv")
    df_mayores_65 = read_csv_url("./data/mayores_65.csv")

    # merging df_morbilidades, df_densidad_poblacional, df_edad_promedio, df_indice_pobreza, df_mayores_65
    df_variables_constantes = merge_dfs_const(
        df_morbilidades,
        df_densidad_poblacional,
        df_edad_promedio,
        df_indice_pobreza,
        df_mayores_65,
    )

    # get constants per day
    df_variables_constantes_dia = get_const_per_day(df_variables_constantes)

    # get data of infected, recovered and deceased using the api of ins
    df_cuenta_departamento_dia = get_ird()

    # merging df_cuenta_departamento_dia with df_variables_constantes_dia
    df_ird_constantes_dia = merge_ird_const(
        df_cuenta_departamento_dia, df_variables_constantes_dia
    )

    # calculate susceptible
    df_sird_constantes_dia = calculate_susceptible(df_ird_constantes_dia)

    # get average morbidity and drop the other morbidity variables
    df_sird_constantes_dia = get_avg_morbidity(df_sird_constantes_dia)

    # write csv
    df_sird_constantes_dia.to_csv("./data/sird_constantes_dia.csv", index=False)


main()
