import pandas as pd


def fit_rf_i(df, pkl_file):

    


# reentrenar cada 7 dias a partir de agosto

# Leer datos
data = "data/Aprendizaje_incremental\data\seird_constantes_dia_colombia_dependencia_cruzada_emulador.csv"
df = pd.read_csv(data)

# Entrenar modelo con los datos hasta Julio
fit_rf_i(df, "./pkl/i_col_emulador.pkl")
