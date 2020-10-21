import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import copy


def predict_s(df, index, filename):
    X = df.loc[
        [index],
        [
            'expuestos_t_5',
            'expuestos_t_6',
            'infectados_t_5',
            'recuperados_t_5',
            'recuperados_t_7'
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "susceptibles_predicho"] = model.predict(X)[0]
    return df


def predict_e(df, index, filename):
    X = df.loc[
        [index],
        [
            'susceptibles_t_6',
            'infectados_t_7',
            'decesos_t_5',
            'decesos_t_7'
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "expuestos_predicho"] = model.predict(X)[0]
    return df


def predict_i(df, index, filename):
    X = df.loc[
        [index],
        [
            'expuestos_t_6',
            'expuestos_t_7',
            'decesos_t_5',
            'decesos_t_6'
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "infectados_predicho"] = model.predict(X)[0]
    return df


def predict_r(df, index, filename):
    X = df.loc[
        [index],
        [
            'susceptibles_t_6',
            'susceptibles_t_7',
            'expuestos_t_5',
            'expuestos_t_6',
            'expuestos_t_7',
            'infectados_t_5',
            'infectados_t_7',
            'decesos_t_5'
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "recuperados_predicho"] = model.predict(X)[0]
    return df


def predict_d(df, index, filename):
    X = df.loc[
        [index],
        [
            'susceptibles_t_5',
            'expuestos_t_5',
            'expuestos_t_6',
            'infectados_t_5',
            'recuperados_t_5',
            'recuperados_t_7',
        ],
    ]

    model = pickle.load(open(filename, "rb"))
    df.loc[index, "decesos_predicho"] = model.predict(X)[0]
    return df


def plot_df(df, y, title):

    graph = df.plot(
        x="days", # date
        y=[y[0], y[2]],
        kind="line",
        figsize=(20, 10),
        grid=True,
        legend=True,
        loglog=False,
        title=title,
        colormap='Dark2_r'
    )
    graph.fill_between(df['days'], df[y[1]], df[y[3]], alpha=0.2)
    plt.ylabel("quantity")
    plt.show()

def get_prediction_interval(df, i, variable, prediction, y_test, test_predictions, pi=.95):
    '''
    Get a prediction interval for a linear regression.
    
    INPUTS: 
        - df
        - index
        - variable: S, E, I, R, D
        - Single prediction, 
        - y_test
        - All test set predictions,
        - Prediction interval threshold (default = .95) 
    OUTPUT: 
        - Prediction interval for single prediction
    '''
    
    #get standard deviation of y_test
    sum_errs = np.sum((test_predictions - y_test)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)

    #get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev

    #generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval
    if lower < 0: lower = 0

    lower = lower * 1.05
    upper = upper * 0.95
    # store results
    if variable == 'S':
        df.loc[i, "susceptibles_lower"] = lower
        df.loc[i, "susceptibles_upper"] = upper
    if variable == 'E':
        df.loc[i, "expuestos_lower"] = lower
        df.loc[i, "expuestos_upper"] = upper
    if variable == 'I':
        df.loc[i, "infectados_lower"] = lower
        df.loc[i, "infectados_upper"] = upper
    if variable == 'R':
        df.loc[i, "recuperados_lower"] = lower
        df.loc[i, "recuperados_upper"] = upper
    if variable == 'D':
        df.loc[i, "decesos_lower"] = lower
        df.loc[i, "decesos_upper"] = upper

    return df


def main():

    df = pd.read_csv(
        "./data/seird_constantes_dia_colombia_ingles.csv")
    df["susceptibles_predicho"] = np.nan
    df["expuestos_predicho"] = np.nan
    df["infectados_predicho"] = np.nan
    df["recuperados_predicho"] = np.nan
    df["decesos_predicho"] = np.nan

    n = df.shape[0]
    m_days_predict = n
    for i in range(n - m_days_predict, n):
        fecha = df.loc[i, "fecha"]
        
        # predicciones
        df = predict_s(df, i, "./pkl/s_colombia_t4.pkl")
        df = predict_e(df, i, "./pkl/e_colombia_t4.pkl")
        df = predict_i(df, i, "./pkl/i_colombia_t4.pkl")
        df = predict_r(df, i, "./pkl/r_colombia_t4.pkl")
        df = predict_d(df, i, "./pkl/d_colombia_t4.pkl")

    # agregar 6 dias mas
    window_forecast = 4
    df_ = copy.deepcopy(df)
    for i in range(window_forecast):
        n = df.shape[0]
        last_date = datetime.strptime(df["fecha"].iloc[[n - 1]].values[0], "%Y-%m-%d")
        last_date = last_date + timedelta(days=1)
        t_5 = df.iloc[[n - 5]]
        t_6 = df.iloc[[n - 6]]
        t_7 = df.iloc[[n - 7]]        
        data = {
            "fecha": last_date.strftime("%Y-%m-%d"),
            "susceptibles_t_5": t_5["susceptibles"].values[0],
            "susceptibles_t_6": t_6["susceptibles"].values[0],
            "susceptibles_t_7": t_7["susceptibles"].values[0],
            "expuestos_t_5": t_5["expuestos"].values[0],
            "expuestos_t_6": t_6["expuestos"].values[0],
            "expuestos_t_7": t_7["expuestos"].values[0],
            "infectados_t_5": t_5["infectados"].values[0],
            "infectados_t_6": t_6["infectados"].values[0],
            "infectados_t_7": t_7["infectados"].values[0],
            "recuperados_t_5": t_5["recuperados"].values[0],
            "recuperados_t_6": t_6["recuperados"].values[0],
            "recuperados_t_7": t_7["recuperados"].values[0],
            "decesos_t_5": t_5["decesos"].values[0],
            "decesos_t_6": t_6["decesos"].values[0],
            "decesos_t_7": t_7["decesos"].values[0],
        }
        df_partial = pd.DataFrame(data, index=[n])
        df = df.append(df_partial)

        # prediccion
        df = predict_s(df, n, "./pkl/s_colombia_t4.pkl")
        df = predict_e(df, n, "./pkl/e_colombia_t4.pkl")
        df = predict_i(df, n, "./pkl/i_colombia_t4.pkl")
        df = predict_r(df, n, "./pkl/r_colombia_t4.pkl")
        df = predict_d(df, n, "./pkl/d_colombia_t4.pkl")

    # intervalos de confianza
    df["susceptibles_lower"] = np.nan
    df["expuestos_lower"] = np.nan
    df["infectados_lower"] = np.nan
    df["recuperados_lower"] = np.nan
    df["decesos_lower"] = np.nan
    df["susceptibles_upper"] = np.nan
    df["expuestos_upper"] = np.nan
    df["infectados_upper"] = np.nan
    df["recuperados_upper"] = np.nan
    df["decesos_upper"] = np.nan
    n = df.shape[0]
    for i in range(n - window_forecast - 1, n):
        df = get_prediction_interval(df, i, 'S', df.loc[i, "susceptibles_predicho"], df_["susceptibles"].tail(window_forecast).to_numpy(), df_["susceptibles_predicho"].tail(window_forecast).to_numpy())
        df = get_prediction_interval(df, i, 'E', df.loc[i, "expuestos_predicho"], df_["expuestos"].tail(window_forecast).to_numpy(), df_["expuestos_predicho"].tail(window_forecast).to_numpy())
        df = get_prediction_interval(df, i, 'I', df.loc[i, "infectados_predicho"], df_["infectados"].tail(window_forecast).to_numpy(), df_["infectados_predicho"].tail(window_forecast).to_numpy())
        df = get_prediction_interval(df, i, 'R', df.loc[i, "recuperados_predicho"], df_["recuperados"].tail(window_forecast).to_numpy(), df_["recuperados_predicho"].tail(window_forecast).to_numpy())
        df = get_prediction_interval(df, i, 'D', df.loc[i, "decesos_predicho"], df_["decesos"].tail(window_forecast).to_numpy(), df_["decesos_predicho"].tail(window_forecast).to_numpy())

    # print(df[['susceptibles_lower', 'susceptibles', 'susceptibles_predicho', 'susceptibles_upper']].tail(35))
    # print(df[['expuestos_lower', 'expuestos', 'expuestos_predicho', 'expuestos_upper']].tail(35))
    # print(df[['infectados_lower', 'infectados', 'infectados_predicho', 'infectados_upper']].tail(35))
    # print(df[['recuperados_lower', 'recuperados', 'recuperados_predicho', 'recuperados_upper']].tail(35))
    # print(df[['decesos_lower', 'decesos', 'decesos_predicho', 'decesos_upper']].tail(35))

    # graficas valores reales vs predichos
    rename_col = {
        'susceptibles': 'susceptible',
        'expuestos': 'exposed',
        'infectados': 'infected',
        'recuperados': 'recovered',
        'decesos': 'deaths',
        'susceptibles_predicho': 'susceptible_predicted',
        'expuestos_predicho': 'exposed_predicted',
        'infectados_predicho': 'infected_predicted',
        'recuperados_predicho': 'recovered_predicted',
        'decesos_predicho': 'deaths_predicted',
        'fecha': 'days'
    }
    df = df.rename(columns=rename_col)
    plot_df(df, ["susceptible",'susceptibles_lower', 'susceptible_predicted', 'susceptibles_upper'], "Confidence interval for prediction - Susceptible")
    plot_df(df, ["exposed", 'expuestos_lower', 'exposed_predicted', 'expuestos_upper'],              "Confidence interval for prediction - Exposed")
    plot_df(df, ["infected", 'infectados_lower', 'infected_predicted', 'infectados_upper'],          "Confidence interval for prediction - Infected")
    plot_df(df, ["recovered", 'recuperados_lower', 'recovered_predicted', 'recuperados_upper'],      "Confidence interval for prediction - Recovered")
    plot_df(df, ["deaths", 'decesos_lower', 'deaths_predicted', 'decesos_upper'],                    "Confidence interval for prediction - Deaths")
    
    return df.round().to_dict(orient="list")


if __name__ == "__main__":
    main()
    # df = main()
    # keys = ['fecha',
    #         'infectados','infectados_predicho',
    #         'decesos','decesos_predicho',
    #         'recuperados','recuperados_predicho',
    #         'susceptibles','susceptibles_predicho',
    #         'expuestos','expuestos_predicho']
    # data = {k:v for k,v in df.items() if k in keys}
    
    # with open('estimations/co.json', 'w') as fp:
    #     json.dump(data, fp)
