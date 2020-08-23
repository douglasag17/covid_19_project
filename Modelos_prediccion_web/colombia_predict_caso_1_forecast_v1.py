import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta, datetime


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
    for i in range(n - 20, n):
        fecha = df.loc[i, "fecha"]
        df = predict_s(df, i, "./pkl/s_colombia_1.pkl")
        df = predict_e(df, i, "./pkl/e_colombia_1.pkl")
        df = predict_i(df, i, "./pkl/i_colombia_1.pkl")
        df = predict_r(df, i, "./pkl/r_colombia_1.pkl")
        df = predict_d(df, i, "./pkl/d_colombia_1.pkl")

    # graficas valores reales vs predichos
    # plot_df(df.iloc[n-20:], ["susceptibles", "susceptibles_predicho"], "susceptibles reales vs susceptibles predicho")
    # plot_df(df.iloc[n-20:], ["expuestos", "expuestos_predicho"], "expuestos reales vs expuestos predicho")
    # plot_df(df.iloc[n-20:], ["infectados", "infectados_predicho"], "infectados reales vs infectados predicho")
    # plot_df(df.iloc[n-20:], ["recuperados", "recuperados_predicho"], "recuperados reales vs recuperados predicho")
    # plot_df(df.iloc[n-20:], ["decesos", "decesos_predicho"], "decesos reales vs decesos predicho")

    # agregar 6 dias mas
    window_forecast = 6
    df.index.name = 'num'
    for i in range(window_forecast):
        n = df.shape[0]
        last_date = datetime.strptime(df['fecha'].iloc[[n-1]].values[0], '%Y-%m-%d')
        last_date = last_date + timedelta(days=1)
        t_1 = df.iloc[[n-1]]
        t_7 = df.iloc[[n-7]]
        if i == 0:
            infectados_t_1 = t_1['infectados'].values[0]
        else:
            infectados_t_1 = t_1['infectados_predicho'].values[0]
        
        data = {
            "fecha": last_date.strftime('%Y-%m-%d'),
            "cantidad_mayores_65": 3994453.0,
            "ipm": 45.759507,
            "poblacion_total": 49598176.0,
            "personas_km2": 52.594,
            "edad_promedio": 33.222,
            "promedio_morbilidades": 4940210.083333333,
            "infectados_t_1": infectados_t_1,
            "infectados_t_7": t_7['infectados'].values[0]
        }
        df_partial = pd.DataFrame(data, index=[n])
        df_partial.index.name = 'num'

        df = df.append(df_partial)

        # prediccion
        df = predict_i(df, n, "./pkl/i_colombia_1.pkl")

    print(df[['fecha', 'infectados', 'infectados_predicho', "infectados_t_1", "infectados_t_7"]].tail(10))

    return df.to_dict(orient="list")


main()




"""

# agregar 6 dias mas
window_forecast = 6
df.index.name = 'num'
df["fecha"] = df["fecha"].astype("datetime64[ns]")
df = df.set_index("fecha", drop=False, append=True, inplace=False)
for i in range(window_forecast):
    n = df.shape[0]
    last_date = df.iloc[[-1]].index.get_level_values('fecha')
    last_date = last_date + timedelta(days=1)
    t_1 = df.loc[[df.iloc[[n-1]].index.get_level_values('num')[-1]]]
    t_7 = df.loc[[df.iloc[[n-7]].index.get_level_values('num')[-1]]]
    data = {
        "num_": n,
        "fecha_": last_date,
        "cantidad_mayores_65": 3994453.0,
        "ipm": 45.759507,
        "poblacion_total": 49598176.0,
        "personas_km2": 52.594,
        "edad_promedio": 33.222,
        "promedio_morbilidades": 4940210.083333333,
        "infectados_t_1": df['infectados_predicho'].iloc[[n-1]],
        "infectados_t_7": t_7['infectados'].values[0]
    }
    df_partial = pd.DataFrame.from_dict(data).reset_index()
    df_partial = df_partial.drop(columns=['num', 'fecha'])
    df_partial = df_partial.rename(columns={"num_": "num", "fecha_": "fecha"})
    df_partial = df_partial.set_index("num", drop=True, append=False, inplace=False)
    df_partial = df_partial.set_index("fecha", drop=False, append=True, inplace=False)

    df = df.append(df_partial)

    # prediccion
    #df.loc[n, "infectados_predicho"] = 1
    df = predict_i(df, n, "./pkl/i_colombia_1.pkl")

print(df[['infectados', 'infectados_predicho', "infectados_t_1", "infectados_t_7"]].tail(10))

"""