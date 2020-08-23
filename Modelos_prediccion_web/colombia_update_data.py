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
    # En INS no esta guania, guaviare, vichada
    df_variables_constantes_1 = df_variables_constantes_1.drop([15, 16, 32])

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
    results = client.get("gt2j-8ykr", limit=10000000)
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


def get_exposed():

    # Expuestos
    client = Socrata("www.datos.gov.co", None)
    results = client.get("8835-5baf", limit=1000000)
    df_expuestos_col = pd.DataFrame.from_records(results)
    df_expuestos_col = df_expuestos_col[["fecha", "acumuladas"]]
    df_expuestos_col = df_expuestos_col.iloc[1:].reset_index(drop=True)
    df_expuestos_col["fecha"] = df_expuestos_col["fecha"].astype(str)
    df_expuestos_col["fecha"] = df_expuestos_col["fecha"].transform(lambda x: x[:10])
    df_expuestos_col["acumuladas"] = df_expuestos_col["acumuladas"].astype(float)
    df_expuestos_col = df_expuestos_col.sort_values(by=["fecha"]).reset_index(drop=True)

    # pasando de acumuladas a por dia
    df_expuestos_col["expuestos"] = 0
    for i, row in df_expuestos_col.iterrows():
        if i == 0:
            df_expuestos_col.loc[i, "expuestos"] = df_expuestos_col.loc[i, "acumuladas"]
        else:
            df_expuestos_col.loc[i, "expuestos"] = (
                df_expuestos_col.loc[i, "acumuladas"]
                - df_expuestos_col.loc[i - 1, "acumuladas"]
            )

    return df_expuestos_col


def get_df_ird_constantes_dia_col(df_ird_constantes_dia, df_variables_constantes_dia):

    # IRD por dia
    df_ird_col = df_ird_constantes_dia.groupby(["fecha"])[
        "infectados", "recuperados", "decesos"
    ].sum()
    df_ird_col.reset_index(level=0, inplace=True)

    # Constantes por dia
    col = df_variables_constantes_dia.columns.to_list()[2:]
    for c in col:
        df_variables_constantes_dia[c] = df_variables_constantes_dia[c].astype(float)

    df_constantes_dia_col = df_variables_constantes_dia.groupby(["fecha"])[col].sum()
    df_constantes_dia_col["ipm"] = 45.759507
    df_constantes_dia_col["personas_km2"] = 52.594
    df_constantes_dia_col["edad_promedio"] = 33.222
    df_constantes_dia_col.reset_index(level=0, inplace=True)

    # Merge df_ird_col con df_constantes_dia_col
    df_ird_constantes_dia_col = df_ird_col.merge(
        df_constantes_dia_col, how="inner", left_on="fecha", right_on="fecha"
    )
    df_ird_constantes_dia_col = df_ird_constantes_dia_col.sort_values(
        by=["fecha"]
    ).reset_index(drop=True)
    return df_ird_constantes_dia_col


def merge_ird_col_e(df_ird_constantes_dia_col, df_expuestos_col):
    df_eird_constantes_dia_col = df_ird_constantes_dia_col.merge(
        df_expuestos_col, how="inner", left_on="fecha", right_on="fecha"
    )
    df_eird_constantes_dia_col = df_eird_constantes_dia_col.drop(columns=["acumuladas"])
    return df_eird_constantes_dia_col


def calculate_susceptible(df_eird_constantes_dia_col):
    df_seird_constantes_dia_col = copy.deepcopy(df_eird_constantes_dia_col)
    df_seird_constantes_dia_col["susceptibles"] = df_seird_constantes_dia_col[
        df_seird_constantes_dia_col["fecha"] == "2020-03-05"
    ]["poblacion_total"]
    df_seird_constantes_dia_col = df_seird_constantes_dia_col.sort_values(
        by=["fecha"]
    ).reset_index(drop=True)
    for i, row in df_seird_constantes_dia_col.iterrows():
        if i > 0:
            df_seird_constantes_dia_col.loc[i, "susceptibles"] = (
                df_seird_constantes_dia_col.loc[i - 1, "susceptibles"]
                - df_seird_constantes_dia_col.loc[i - 1, "expuestos"]
                - df_seird_constantes_dia_col.loc[i - 1, "infectados"]
                - df_seird_constantes_dia_col.loc[i - 1, "recuperados"]
                - df_seird_constantes_dia_col.loc[i - 1, "decesos"]
            )

    # Ordenar columnas
    df_seird_constantes_dia_col = df_seird_constantes_dia_col[
        [
            "fecha",
            "susceptibles",
            "expuestos",
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
    return df_seird_constantes_dia_col


def get_avg_morbidity(df_seird_constantes_dia_col):

    df_seird_constantes_dia_col_ = copy.deepcopy(df_seird_constantes_dia_col)
    df_seird_constantes_dia_col_["promedio_morbilidades"] = (
        df_seird_constantes_dia_col_["anomalias_congenitas"]
        + df_seird_constantes_dia_col_["codiciones_orales"]
        + df_seird_constantes_dia_col_[
            "condiciones_derivadas_durante_el_periodo_perinatal"
        ]
        + df_seird_constantes_dia_col_["condiciones_maternas"]
        + df_seird_constantes_dia_col_["condiciones_neuropsiquiatricas"]
        + df_seird_constantes_dia_col_["deficiencias_nutricionales"]
        + df_seird_constantes_dia_col_["desordenes_endocrinos"]
        + df_seird_constantes_dia_col_["diabetes_mellitus"]
        + df_seird_constantes_dia_col_["enfermedades_cardiovasculares"]
        + df_seird_constantes_dia_col_["enfermedades_de_la_piel"]
        + df_seird_constantes_dia_col_["enfermedades_de_los_organos_de_los_sentidos"]
        + df_seird_constantes_dia_col_["enfermedades_digestivas"]
        + df_seird_constantes_dia_col_["enfermedades_genitourinarias"]
        + df_seird_constantes_dia_col_["enfermedades_infecciosas_y_parasitarias"]
        + df_seird_constantes_dia_col_["enfermedades_musculo_esqueleticas"]
        + df_seird_constantes_dia_col_["enfermedades_respiratorias"]
        + df_seird_constantes_dia_col_["infecciones_respiratorias"]
        + df_seird_constantes_dia_col_["lesiones_de_intensionalidad_indeterminada"]
        + df_seird_constantes_dia_col_["lesiones_intensionales"]
        + df_seird_constantes_dia_col_["lesiones_no_intensionales"]
        + df_seird_constantes_dia_col_["neoplasias_malignas"]
        + df_seird_constantes_dia_col_["otras_neoplasias"]
        + df_seird_constantes_dia_col_["signos_y_sintomas_mal_definidos"]
        + df_seird_constantes_dia_col_[
            "traumatismos__envenenamientos_u_algunas_otras_consecuencias_de_causas_externas"
        ]
    ) / 24

    df_seird_constantes_dia_col_ = df_seird_constantes_dia_col_[
        [
            "fecha",
            "susceptibles",
            "expuestos",
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

    # seird t-1 y t-1
    df_seird_constantes_dia_col_["susceptibles_t_1"] = None
    df_seird_constantes_dia_col_["susceptibles_t_2"] = None
    df_seird_constantes_dia_col_["susceptibles_t_3"] = None
    df_seird_constantes_dia_col_["susceptibles_t_4"] = None
    df_seird_constantes_dia_col_["susceptibles_t_5"] = None
    df_seird_constantes_dia_col_["susceptibles_t_6"] = None
    df_seird_constantes_dia_col_["susceptibles_t_7"] = None
    df_seird_constantes_dia_col_["expuestos_t_1"] = None
    df_seird_constantes_dia_col_["expuestos_t_2"] = None
    df_seird_constantes_dia_col_["expuestos_t_3"] = None
    df_seird_constantes_dia_col_["expuestos_t_4"] = None
    df_seird_constantes_dia_col_["expuestos_t_5"] = None
    df_seird_constantes_dia_col_["expuestos_t_6"] = None
    df_seird_constantes_dia_col_["expuestos_t_7"] = None
    df_seird_constantes_dia_col_["infectados_t_1"] = None
    df_seird_constantes_dia_col_["infectados_t_2"] = None
    df_seird_constantes_dia_col_["infectados_t_3"] = None
    df_seird_constantes_dia_col_["infectados_t_4"] = None
    df_seird_constantes_dia_col_["infectados_t_5"] = None
    df_seird_constantes_dia_col_["infectados_t_6"] = None
    df_seird_constantes_dia_col_["infectados_t_7"] = None
    df_seird_constantes_dia_col_["recuperados_t_1"] = None
    df_seird_constantes_dia_col_["recuperados_t_2"] = None
    df_seird_constantes_dia_col_["recuperados_t_3"] = None
    df_seird_constantes_dia_col_["recuperados_t_4"] = None
    df_seird_constantes_dia_col_["recuperados_t_5"] = None
    df_seird_constantes_dia_col_["recuperados_t_6"] = None
    df_seird_constantes_dia_col_["recuperados_t_7"] = None
    df_seird_constantes_dia_col_["decesos_t_1"] = None
    df_seird_constantes_dia_col_["decesos_t_2"] = None
    df_seird_constantes_dia_col_["decesos_t_3"] = None
    df_seird_constantes_dia_col_["decesos_t_4"] = None
    df_seird_constantes_dia_col_["decesos_t_5"] = None
    df_seird_constantes_dia_col_["decesos_t_6"] = None
    df_seird_constantes_dia_col_["decesos_t_7"] = None
    for i, row in df_seird_constantes_dia_col_.iterrows():
        if i > 7:
            df_seird_constantes_dia_col_.loc[
                i, "susceptibles_t_1"
            ] = df_seird_constantes_dia_col_.loc[i - 1, "susceptibles"]
            df_seird_constantes_dia_col_.loc[
                i, "susceptibles_t_2"
            ] = df_seird_constantes_dia_col_.loc[i - 2, "susceptibles"]
            df_seird_constantes_dia_col_.loc[
                i, "susceptibles_t_3"
            ] = df_seird_constantes_dia_col_.loc[i - 3, "susceptibles"]
            df_seird_constantes_dia_col_.loc[
                i, "susceptibles_t_4"
            ] = df_seird_constantes_dia_col_.loc[i - 4, "susceptibles"]
            df_seird_constantes_dia_col_.loc[
                i, "susceptibles_t_5"
            ] = df_seird_constantes_dia_col_.loc[i - 5, "susceptibles"]
            df_seird_constantes_dia_col_.loc[
                i, "susceptibles_t_6"
            ] = df_seird_constantes_dia_col_.loc[i - 6, "susceptibles"]
            df_seird_constantes_dia_col_.loc[
                i, "susceptibles_t_7"
            ] = df_seird_constantes_dia_col_.loc[i - 7, "susceptibles"]
            df_seird_constantes_dia_col_.loc[
                i, "expuestos_t_1"
            ] = df_seird_constantes_dia_col_.loc[i - 1, "expuestos"]
            df_seird_constantes_dia_col_.loc[
                i, "expuestos_t_2"
            ] = df_seird_constantes_dia_col_.loc[i - 2, "expuestos"]
            df_seird_constantes_dia_col_.loc[
                i, "expuestos_t_3"
            ] = df_seird_constantes_dia_col_.loc[i - 3, "expuestos"]
            df_seird_constantes_dia_col_.loc[
                i, "expuestos_t_4"
            ] = df_seird_constantes_dia_col_.loc[i - 4, "expuestos"]
            df_seird_constantes_dia_col_.loc[
                i, "expuestos_t_5"
            ] = df_seird_constantes_dia_col_.loc[i - 5, "expuestos"]
            df_seird_constantes_dia_col_.loc[
                i, "expuestos_t_6"
            ] = df_seird_constantes_dia_col_.loc[i - 6, "expuestos"]
            df_seird_constantes_dia_col_.loc[
                i, "expuestos_t_7"
            ] = df_seird_constantes_dia_col_.loc[i - 7, "expuestos"]
            df_seird_constantes_dia_col_.loc[
                i, "infectados_t_1"
            ] = df_seird_constantes_dia_col_.loc[i - 1, "infectados"]
            df_seird_constantes_dia_col_.loc[
                i, "infectados_t_2"
            ] = df_seird_constantes_dia_col_.loc[i - 2, "infectados"]
            df_seird_constantes_dia_col_.loc[
                i, "infectados_t_3"
            ] = df_seird_constantes_dia_col_.loc[i - 3, "infectados"]
            df_seird_constantes_dia_col_.loc[
                i, "infectados_t_4"
            ] = df_seird_constantes_dia_col_.loc[i - 4, "infectados"]
            df_seird_constantes_dia_col_.loc[
                i, "infectados_t_5"
            ] = df_seird_constantes_dia_col_.loc[i - 5, "infectados"]
            df_seird_constantes_dia_col_.loc[
                i, "infectados_t_6"
            ] = df_seird_constantes_dia_col_.loc[i - 6, "infectados"]
            df_seird_constantes_dia_col_.loc[
                i, "infectados_t_7"
            ] = df_seird_constantes_dia_col_.loc[i - 7, "infectados"]
            df_seird_constantes_dia_col_.loc[
                i, "recuperados_t_1"
            ] = df_seird_constantes_dia_col_.loc[i - 1, "recuperados"]
            df_seird_constantes_dia_col_.loc[
                i, "recuperados_t_2"
            ] = df_seird_constantes_dia_col_.loc[i - 2, "recuperados"]
            df_seird_constantes_dia_col_.loc[
                i, "recuperados_t_3"
            ] = df_seird_constantes_dia_col_.loc[i - 3, "recuperados"]
            df_seird_constantes_dia_col_.loc[
                i, "recuperados_t_4"
            ] = df_seird_constantes_dia_col_.loc[i - 4, "recuperados"]
            df_seird_constantes_dia_col_.loc[
                i, "recuperados_t_5"
            ] = df_seird_constantes_dia_col_.loc[i - 5, "recuperados"]
            df_seird_constantes_dia_col_.loc[
                i, "recuperados_t_6"
            ] = df_seird_constantes_dia_col_.loc[i - 6, "recuperados"]
            df_seird_constantes_dia_col_.loc[
                i, "recuperados_t_7"
            ] = df_seird_constantes_dia_col_.loc[i - 7, "recuperados"]
            df_seird_constantes_dia_col_.loc[
                i, "decesos_t_1"
            ] = df_seird_constantes_dia_col_.loc[i - 1, "decesos"]
            df_seird_constantes_dia_col_.loc[
                i, "decesos_t_2"
            ] = df_seird_constantes_dia_col_.loc[i - 2, "decesos"]
            df_seird_constantes_dia_col_.loc[
                i, "decesos_t_3"
            ] = df_seird_constantes_dia_col_.loc[i - 3, "decesos"]
            df_seird_constantes_dia_col_.loc[
                i, "decesos_t_4"
            ] = df_seird_constantes_dia_col_.loc[i - 4, "decesos"]
            df_seird_constantes_dia_col_.loc[
                i, "decesos_t_5"
            ] = df_seird_constantes_dia_col_.loc[i - 5, "decesos"]
            df_seird_constantes_dia_col_.loc[
                i, "decesos_t_6"
            ] = df_seird_constantes_dia_col_.loc[i - 6, "decesos"]
            df_seird_constantes_dia_col_.loc[
                i, "decesos_t_7"
            ] = df_seird_constantes_dia_col_.loc[i - 7, "decesos"]

    df_seird_constantes_dia_col_ = df_seird_constantes_dia_col_.dropna(axis=0)
    return df_seird_constantes_dia_col_


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

    # get expuestos
    df_expuestos_col = get_exposed()

    # df_ird_constantes_dia para Colombia
    df_ird_constantes_dia_col = get_df_ird_constantes_dia_col(
        df_ird_constantes_dia, df_variables_constantes_dia
    )

    # Merge df_ird_constantes_dia_col con df_expuestos_col
    df_eird_constantes_dia_col = merge_ird_col_e(
        df_ird_constantes_dia_col, df_expuestos_col
    )

    # calculate susceptible
    df_seird_constantes_dia_col = calculate_susceptible(df_eird_constantes_dia_col)

    # get average morbidity and drop the other morbidity variables
    df_seird_constantes_dia_col_ = get_avg_morbidity(df_seird_constantes_dia_col)

    # write csv
    df_seird_constantes_dia_col_.to_csv(
        "./data/seird_constantes_dia_colombia_dependencia_cruzada.csv", index=False
    )


main()
