import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, date
import matplotlib.pyplot as plt


def get_trends_colombia(pytrend, kw_list, current_date, dest_file):

    pytrend.build_payload(
        kw_list=kw_list,
        cat=0,
        timeframe=f"2020-02-01 {current_date}",
        geo="CO",
        gprop="",
    )

    # Interest over time in Colombia
    df_col = pytrend.interest_over_time()
    df_col = df_col.reset_index()
    df_col = df_col.drop(labels=["isPartial"], axis="columns")
    df_col.to_csv(dest_file, index=False)

    df_col.plot(x='date', y=kw_list, figsize=(120, 10), kind ='line')
    plt.show()


def get_trends_departments(pytrend, kw_list, current_date, dest_file):

    # Interest over time by department
    departments = {
        "amazonas": "AMA",
        "antioquia": "ANT",
        "arauca": "ARA",
        "atlantico": "ATL",
        "bogota": "CUN",
        "bolivar": "BOL",
        "boyaca": "BOY",
        "caldas": "CAL",
        "caqueta": "CAQ",
        "casanare": "CAS",
        "cauca": "CAU",
        "cesar": "CES",
        "choco": "CHO",
        "cordoba": "COR",
        "cundinamarca": "CUN",
        "huila": "HUI",
        "la guajira": "LAG",
        "magdalena": "MAG",
        "meta": "MET",
        "narino": "NAR",
        "norte de santander": "",
        "putumayo": "PUT",
        "quindio": "",
        "risaralda": "",
        "san andres y providencia": "SAP",
        "santander": "SAN",
        "sucre": "SUC",
        "tolima": "TOL",
        "valle del cauca": "VAC",
        "vaupes": "VAU"
    }
    
    dfs = []
    for dep, code in departments.items():
        if code != "":
            try:
                geo = f"CO-{code}"
                pytrend.build_payload(
                    kw_list=kw_list,
                    cat=0,
                    timeframe=f"2020-02-01 {current_date}",
                    geo=geo,
                    gprop="",
                )
            except:
                raise Exception(f"fail {geo}")
            else:
                df_dep = pytrend.interest_over_time()
                df_dep = df_dep.reset_index()
                df_dep = df_dep.drop(labels=["isPartial"], axis="columns")
                df_dep["department"] = dep
                dfs.append(df_dep)

    df_departments = pd.concat(dfs, axis=0)
    df_departments = df_departments.sort_values(by=["date"]).reset_index(drop=True)
    df_departments.to_csv(dest_file, index=False)

if __name__ == "__main__":
    pytrend = TrendReq()
    kw_list = ["coronavirus", "covid", "cuarentena"]
    current_date = datetime.now().date().strftime("%Y-%m-%d")
    get_trends_colombia(pytrend, kw_list, current_date, "google_trends_colombia.csv")
    get_trends_departments(pytrend, kw_list, current_date, "google_trends_departamentos.csv")

