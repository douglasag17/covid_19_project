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
    print(cluster_labels)

    df.loc[:, 'cluster'] = cluster_labels

    df['departamento'] = df.index
    res = df[["departamento", "cluster"]].to_dict("records")
    return res

if __name__ == "__main__":
    res = main()
    print(res)
