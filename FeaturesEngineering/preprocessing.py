import pandas as pd
import numpy as np
import time
import datetime 
from datetime import datetime

def prepocess_data(path):
    df=pd.read_csv(path)

    date2year = lambda d : datetime.strptime(d, "%d/%m/%Y").year
    def days_between(d1, d2):
        d1 = datetime.strptime(d1, "%d/%m/%Y")
        d2 = datetime.strptime(d2, "%d/%m/%Y")
        return abs((d2 - d1).days)


    df["join_age"] = df.Customer_Open_Date.apply(date2year)-df.BirthDate.apply(date2year)
    df["sub_delay"] = df.apply(lambda x: days_between(x.Prod_Decision_Date, x.Customer_Open_Date), axis=1)
    df["is_closed_date"]= 1 - df.Prod_Closed_Date.isnull()*1
    df.Net_Annual_Income = df.Net_Annual_Income.replace({',': '.'}, regex=True).astype(float).apply(np.floor)
    df = df.drop("BirthDate", axis=1)
    df = df.drop("Customer_Open_Date", axis=1)
    df = df.drop("Prod_Decision_Date", axis=1)
    df = df.drop("Prod_Closed_Date", axis=1)
    df = df.drop("Id_Customer", axis=1)

    df = df.dropna()
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get data path")
    parser.add_argument("--path", action="store", help="path to data")
    args = parser.parse_args()
    
    df = prepocess_data(args.path)
    print(df)