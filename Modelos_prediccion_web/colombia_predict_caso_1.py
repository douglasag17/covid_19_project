import pandas as pd
import pickle
import matplotlib.pyplot as plt


def predict_s(df, index, filename):
    X = df.loc[
        [index],
        [
            "recuperados_t_1",
            "infectados_t_1",
            "decesos_t_1",
            "personas_km2",
            "ipm",
            "promedio_morbilidades",
            "edad_promedio",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "susceptibles_predicho"] = model.predict(X)[0]
    return df


def predict_e(df, index, filename):
    X = df.loc[
        [index], ["expuestos_t_1", "personas_km2", "ipm", "promedio_morbilidades"],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "expuestos_predicho"] = model.predict(X)[0]
    return df


def predict_i(df, index, filename):
    X = df.loc[
        [index],
        [
            "infectados_t_1",
            "infectados_t_7",
            "personas_km2",
            "promedio_morbilidades",
            "edad_promedio",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "infectados_predicho"] = model.predict(X)[0]
    return df


def predict_r(df, index, filename):
    X = df.loc[
        [index],
        [
            "recuperados_t_1",
            "recuperados_t_7",
            "personas_km2",
            "promedio_morbilidades",
            "edad_promedio",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "recuperados_predicho"] = model.predict(X)[0]
    return df


def predict_d(df, index, filename):
    X = df.loc[
        [index],
        [
            "decesos_t_1",
            "decesos_t_7",
            "personas_km2",
            "promedio_morbilidades",
            "edad_promedio",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "decesos_predicho"] = model.predict(X)[0]
    return df


def plot_df(df, y, title):
    graph = df.plot(
        x="fecha",
        y=y,
        kind="line",
        figsize=(20, 10),
        grid=True,
        legend=True,
        loglog=False,
        title=title,
    )
    plt.ylabel("quantity")
    plt.show()


def main():

    df = pd.read_csv("./data/seird_constantes_dia_colombia_dependencia_cruzada.csv")
    df["susceptibles_predicho"] = df["susceptibles"]
    df["expuestos_predicho"] = df["expuestos"]
    df["infectados_predicho"] = df["infectados"]
    df["recuperados_predicho"] = df["recuperados"]
    df["decesos_predicho"] = df["decesos"]

    n = df.shape[0]
    for i in range(n - n, n):
        fecha = df.loc[i, "fecha"]
        df = predict_s(df, i, "./pkl/s_colombia_1.pkl")
        df = predict_e(df, i, "./pkl/e_colombia_1.pkl")
        df = predict_i(df, i, "./pkl/i_colombia_1.pkl")
        df = predict_r(df, i, "./pkl/r_colombia_1.pkl")
        df = predict_d(df, i, "./pkl/d_colombia_1.pkl")

    # organizar datos para graficar en pagina web
    df = df[
        [
            "fecha",
            "susceptibles",
            "susceptibles_predicho",
            "expuestos",
            "expuestos_predicho",
            "infectados",
            "infectados_predicho",
            "recuperados",
            "recuperados_predicho",
            "decesos",
            "decesos_predicho",
        ]
    ]

    # graficas valores reales vs predichos
    plot_df(df.iloc[75:99], ["susceptibles", "susceptibles_predicho"], "susceptibles reales vs susceptibles predicho")
    plot_df(df.iloc[75:99], ["expuestos", "expuestos_predicho"], "expuestos reales vs expuestos predicho")
    plot_df(df.iloc[75:99], ["infectados", "infectados_predicho"], "infectados reales vs infectados predicho")
    plot_df(df.iloc[75:99], ["recuperados", "recuperados_predicho"], "recuperados reales vs recuperados predicho")
    plot_df(df.iloc[75:99], ["decesos", "decesos_predicho"], "decesos reales vs decesos predicho")

    df["fecha"] = df["fecha"].astype("datetime64[ns]")
    df.set_index("fecha", inplace=True)
    return df.to_dict(orient="list")


main()
