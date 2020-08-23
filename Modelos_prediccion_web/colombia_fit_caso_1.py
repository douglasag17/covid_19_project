import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    explained_variance_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
import pickle


def normalize(df):
    df = df.drop(columns=["fecha"])
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df)
    df.loc[:, :] = scaled_values
    return df


def randomForestRegressor(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # n = y.shape[0]
    # lim = 15
    # X_train, X_test, y_train, y_test = X.iloc[:n-lim], X.iloc[n-lim:], y.iloc[:n-lim], y.iloc[n-lim:]
    regr = RandomForestRegressor(
        n_estimators=100,
        criterion="mse",
        max_depth=100,
        min_samples_leaf=2,
        bootstrap=True,
        warm_start=False,
    )
    print(f"Entrenando {y.name}")
    regr.fit(X_train, y_train)
    predicts = regr.predict(X_test)

    # print("Parameters:", regr.get_params())
    # print("Mean Absolute Error:", mean_absolute_error(y_test, predicts))
    # print("Mean Squared Error:", mean_squared_error(y_test, predicts))
    # print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, predicts)))
    # print("Mean Absolute Percentage Error:", np.mean(np.abs((y_test, predicts)) * 100))
    print("R2:", regr.score(X_test, y_test))  # , r2_score(y_test, predicts))

    return regr


def fit_s_1(df, filename):
    X_s = df[
        [
            "recuperados_t_1",
            "infectados_t_1",
            "decesos_t_1",
            "personas_km2",
            "ipm",
            "promedio_morbilidades",
            "edad_promedio",
        ]
    ]
    y_s = df["susceptibles"]
    res_rf_s = randomForestRegressor(X_s, y_s)
    pickle.dump(res_rf_s, open(filename, "wb"))


def fit_e_1(df, filename):
    X_e = df[["expuestos_t_1", "personas_km2", "ipm", "promedio_morbilidades"]]
    y_e = df["expuestos"]
    res_rf_e = randomForestRegressor(X_e, y_e)
    pickle.dump(res_rf_e, open(filename, "wb"))


def fit_i_1(df, filename):
    X_i = df[
        [
            "infectados_t_1",
            "infectados_t_7",
            "personas_km2",
            "promedio_morbilidades",
            "edad_promedio",
        ]
    ]
    y_i = df["infectados"]
    res_rf_i = randomForestRegressor(X_i, y_i)
    pickle.dump(res_rf_i, open(filename, "wb"))


def fit_r_1(df, filename):
    X_r = df[
        [
            "recuperados_t_1",
            "recuperados_t_7",
            "personas_km2",
            "promedio_morbilidades",
            "edad_promedio",
        ]
    ]
    y_r = df["recuperados"]
    res_rf_r = randomForestRegressor(X_r, y_r)
    pickle.dump(res_rf_r, open(filename, "wb"))


def fit_d_1(df, filename):
    X_d = df[
        [
            "decesos_t_1",
            "decesos_t_7",
            "personas_km2",
            "promedio_morbilidades",
            "edad_promedio",
        ]
    ]
    y_d = df["decesos"]
    res_rf_d = randomForestRegressor(X_d, y_d)
    pickle.dump(res_rf_d, open(filename, "wb"))


def main():
    df = pd.read_csv("./data/seird_constantes_dia_colombia_dependencia_cruzada.csv")
    # df = normalize(df)

    # Susceptible
    fit_s_1(df, "./pkl/s_colombia_1.pkl")

    # Expuestos
    fit_e_1(df, "./pkl/e_colombia_1.pkl")

    # Infectados
    fit_i_1(df, "./pkl/i_colombia_1.pkl")

    # Recuperados
    fit_r_1(df, "./pkl/r_colombia_1.pkl")

    # Decesos
    fit_d_1(df, "./pkl/d_colombia_1.pkl")


main()
