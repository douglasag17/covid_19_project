import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np


def predict_s(df, index, filename):
    X = df.loc[
        [index],
        [
            "susceptibles_t_1",
            "expuestos_t_4",
            "expuestos_t_5",
            "expuestos_t_7",
            "infectados_t_1",
            "infectados_t_2",
            "infectados_t_3",
            "infectados_t_4",
            "infectados_t_5",
            "infectados_t_6",
            "infectados_t_7",
            # "recuperados",
            "recuperados_t_1",
            "recuperados_t_2",
            "recuperados_t_7",
            # "decesos",
            "decesos_t_1",
            "decesos_t_2",
            "decesos_t_4",
            "decesos_t_6",
            "decesos_t_7",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "susceptibles_predicho"] = model.predict(X)[0]
    return df


def predict_e(df, index, filename):
    X = df.loc[
        [index],
        [
            # "susceptibles",
            "susceptibles_t_1",
            "susceptibles_t_3",
            "susceptibles_t_5",
            # "infectados",
            "infectados_t_1",
            "infectados_t_6",
            "infectados_t_7",
            # "recuperados",
            "recuperados_t_4",
            "decesos_t_4",
            "decesos_t_6",
            "decesos_t_7",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "expuestos_predicho"] = model.predict(X)[0]
    return df


def predict_i(df, index, filename):
    X = df.loc[
        [index],
        [
            "infectados_t_1",
            "infectados_t_2",
            "susceptibles_t_3",
            "susceptibles_t_5",
            "susceptibles_t_6",
            "expuestos_t_1",
            "expuestos_t_2",
            "expuestos_t_5",
            # "recuperados",
            "recuperados_t_1",
            "recuperados_t_2",
            "recuperados_t_4",
            "recuperados_t_7",
            # "decesos",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "infectados_predicho"] = model.predict(X)[0]
    return df


def predict_r(df, index, filename):
    X = df.loc[
        [index],
        [
            # "susceptibles",
            "susceptibles_t_3",
            "susceptibles_t_4",
            "susceptibles_t_6",
            # "expuestos",
            "expuestos_t_3",
            "expuestos_t_5",
            "expuestos_t_6",
            "infectados_t_1",
            "infectados_t_2",
            "infectados_t_3",
            "decesos_t_5",
            "decesos_t_6",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "recuperados_predicho"] = model.predict(X)[0]
    return df


def predict_d(df, index, filename):
    X = df.loc[
        [index],
        [
            "susceptibles_t_2",
            "susceptibles_t_3",
            "susceptibles_t_7",
            # "expuestos",
            "expuestos_t_2",
            "expuestos_t_3",
            "expuestos_t_5",
            "expuestos_t_7",
            "infectados_t_1",
            "infectados_t_3",
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "decesos_predicho"] = model.predict(X)[0]
    return df


def plot_df(df, y, title):

    graph = df.plot(
        x="Date",
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
    df["susceptibles_predicho"] = np.nan
    df["expuestos_predicho"] = np.nan
    df["infectados_predicho"] = np.nan
    df["recuperados_predicho"] = np.nan
    df["decesos_predicho"] = np.nan

    n_ = df.shape[0]
    for i in range(n_ - n_, n_):
        fecha = df.loc[i, "fecha"]
        df = predict_s(df, i, "./pkl/s_colombia_2.pkl")
        df = predict_e(df, i, "./pkl/e_colombia_2.pkl")
        df = predict_i(df, i, "./pkl/i_colombia_2.pkl")
        df = predict_r(df, i, "./pkl/r_colombia_2.pkl")
        df = predict_d(df, i, "./pkl/d_colombia_2.pkl")

    # agregar 6 dias mas
    window_forecast = 6
    for i in range(window_forecast):
        n = df.shape[0]
        last_date = datetime.strptime(df["fecha"].iloc[[n - 1]].values[0], "%Y-%m-%d")
        last_date = last_date + timedelta(days=1)
        t_1 = df.iloc[[n - 1]]
        t_2 = df.iloc[[n - 2]]
        t_3 = df.iloc[[n - 3]]
        t_4 = df.iloc[[n - 4]]
        t_5 = df.iloc[[n - 5]]
        t_6 = df.iloc[[n - 6]]
        t_7 = df.iloc[[n - 7]]
        data = {
            "fecha": last_date.strftime("%Y-%m-%d"),
            "susceptibles_t_1": t_1["susceptibles_predicho"].values[0],
            "susceptibles_t_2": t_2["susceptibles_predicho"].values[0],
            "susceptibles_t_3": t_3["susceptibles_predicho"].values[0],
            "susceptibles_t_4": t_4["susceptibles_predicho"].values[0],
            "susceptibles_t_5": t_5["susceptibles_predicho"].values[0],
            "susceptibles_t_6": t_6["susceptibles_predicho"].values[0],
            "susceptibles_t_7": t_7["susceptibles_predicho"].values[0],
            "expuestos_t_1": t_1["expuestos_predicho"].values[0],
            "expuestos_t_2": t_2["expuestos_predicho"].values[0],
            "expuestos_t_3": t_3["expuestos_predicho"].values[0],
            "expuestos_t_4": t_4["expuestos_predicho"].values[0],
            "expuestos_t_5": t_5["expuestos_predicho"].values[0],
            "expuestos_t_6": t_6["expuestos_predicho"].values[0],
            "expuestos_t_7": t_7["expuestos_predicho"].values[0],
            "infectados_t_1": t_1["infectados_predicho"].values[0],
            "infectados_t_2": t_2["infectados_predicho"].values[0],
            "infectados_t_3": t_3["infectados_predicho"].values[0],
            "infectados_t_4": t_4["infectados_predicho"].values[0],
            "infectados_t_5": t_5["infectados_predicho"].values[0],
            "infectados_t_6": t_6["infectados_predicho"].values[0],
            "infectados_t_7": t_7["infectados_predicho"].values[0],
            "recuperados_t_1": t_1["recuperados_predicho"].values[0],
            "recuperados_t_2": t_2["recuperados_predicho"].values[0],
            "recuperados_t_3": t_3["recuperados_predicho"].values[0],
            "recuperados_t_4": t_4["recuperados_predicho"].values[0],
            "recuperados_t_5": t_5["recuperados_predicho"].values[0],
            "recuperados_t_6": t_6["recuperados_predicho"].values[0],
            "recuperados_t_7": t_7["recuperados_predicho"].values[0],
            "decesos_t_1": t_1["decesos_predicho"].values[0],
            "decesos_t_2": t_2["decesos_predicho"].values[0],
            "decesos_t_3": t_3["decesos_predicho"].values[0],
            "decesos_t_4": t_4["decesos_predicho"].values[0],
            "decesos_t_5": t_5["decesos_predicho"].values[0],
            "decesos_t_6": t_6["decesos_predicho"].values[0],
            "decesos_t_7": t_7["decesos_predicho"].values[0],
        }
        df_partial = pd.DataFrame(data, index=[n])
        df = df.append(df_partial)

        # prediccion
        df = predict_s(df, n, "./pkl/s_colombia_2.pkl")
        df = predict_e(df, n, "./pkl/e_colombia_2.pkl")
        df = predict_i(df, n, "./pkl/i_colombia_2.pkl")
        df = predict_r(df, n, "./pkl/r_colombia_2.pkl")
        df = predict_d(df, n, "./pkl/d_colombia_2.pkl")

        # reentrenar
        X = df.loc[n - 7 : n][
            [
                "infectados_t_1",
                "infectados_t_2",
                "susceptibles_t_3",
                "susceptibles_t_5",
                "susceptibles_t_6",
                "expuestos_t_1",
                "expuestos_t_2",
                "expuestos_t_5",
                "recuperados_t_1",
                "recuperados_t_2",
                "recuperados_t_4",
                "recuperados_t_7",
            ]
        ]
        y = df.loc[n - 7 : n]["infectados_predicho"]
        print(y)
        model = pickle.load(open("./pkl/i_colombia_2.pkl", "rb"))
        model.partial_fit(X, y)
        pickle.dump(model, open("./pkl/i_colombia_2.pkl", 'wb'))

    # print(
    #     df[
    #         [
    #             "fecha",
    #             "susceptibles_predicho",
    #             "expuestos_predicho",
    #             "infectados_predicho",
    #             "recuperados_predicho",
    #             "decesos_predicho",
    #         ]
    #     ].tail(10)
    # )

    # cambio de nombre
    df = df.rename(
        columns={
            "infectados": "Infected reported",
            "infectados_predicho": "Infected predicted",
            "fecha": "Date",
        }
    )

    # graficas valores reales vs predichos
    # plot_df(df, ["susceptibles", "susceptibles_predicho"], "susceptibles reales vs susceptibles predicho")
    # plot_df(df, ["expuestos", "expuestos_predicho"], "expuestos reales vs expuestos predicho")
    plot_df(
        df,
        ["Infected reported", "Infected predicted"],
        "Infected reported vs Infected predicted",
    )
    # plot_df(df, ["recuperados", "recuperados_predicho"], "recuperados reales vs recuperados predicho")
    # plot_df(df, ["decesos", "decesos_predicho"], "decesos reales vs decesos predicho")
    return df.to_dict(orient="list")


main()
