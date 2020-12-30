# Esto se ejecuta cada mes (el dia 1 de cada mes)

import pandas as pd
from sklearn.cluster import KMeans


def get_data(path):
    # solo usar variables resultantes de la reduccion de dimension
    variables_cluster = [
        "departamento",
        "inc_pred",
        "ipm",
        "('Indicador de barreras a servicios para cuidado de la primera infancia en el total del municipio', 'BSCPI_TOT')",
        "('Indicador de hacinamiento crítico en los centros poblados y el rural disperso', 'HC_CPRD')",
        "('Indicador de inasistencia escolar en las cabeceras municipales', 'IE_CAB')",
        "('Indicador de rezago escolar en los centros poblados y el rural disperso', 'RE_CPRD')",
        "('Indicador de trabajo informal en las cabeceras municipales', 'TRIML_CAB')",
        "% DE MUJERES EN EL DEPARTAMENTO",
        "% DE PERSONAS EN EL DEPARTAMENTO15_64",
    ]
    df = pd.read_csv(path, usecols=variables_cluster)
    df = df.set_index(df["departamento"], drop=True)
    df = df.drop(columns=["departamento"])

    normalized_df = (df - df.mean()) / df.std()
    normalized_df = normalized_df.fillna(0)
    return normalized_df


def kmeans_fit(df):
    clust = KMeans(n_clusters=3, random_state=42)
    cluster_labels = clust.fit_predict(df)
    return list(cluster_labels)


def main():
    df = get_data("./data/sird_dia_embeddings.csv")

    # fit kmeans
    cluster_labels = kmeans_fit(df)
    # print(cluster_labels)

    df.loc[:, 'cluster'] = cluster_labels

    df['departamento'] = df.index

    codigo_departamentos = {
        'amazonas': '91',
        'antioquia': '05',
        'arauca': '81',
        'atlantico': '08',
        'bogota': '11',
        'bolivar': '13',
        'boyaca': '15',
        'caldas': '17',
        'caqueta': '18',
        'casanare': '85',
        'cauca': '19',
        'cesar': '20',
        'choco': '27',
        'cordoba': '23',    
        'cundinamarca': '25',
        'guainia': '94',
        'guaviare': '95',
        'huila': '41',
        'la guajira': '44',
        'magdalena': '47',
        'meta': '50',
        'narino': '52',
        'norte de santander': '54',
        'putumayo': '86',
        'quindio': '63',
        'risaralda': '66',
        'san andres y providencia': '88',
        'santander': '68',
        'sucre': '70',
        'tolima': '73',
        'valle del cauca': '76',
        'vaupes': '97',
        'vichada': '99'
    }

    data = df[["departamento", "cluster"]].to_dict()['cluster']
    res = {}
    for k, v in data.items():
        res[codigo_departamentos[k]] = v

    return res

if __name__ == "__main__":
    res = main()
    print(res)
