import pandas as pd
import numpy as np
import time
import datetime 
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append("../../CreditDefaultPrediction")

from utils.memory import *

def prepocess_data(path):
    """path
    
    Arguments:
        path {string} -- path to data
    
    Returns:
        dataframe -- containt data preprocessed with a feature engineering done
    """
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
    ##Features engeenering 

    ## Income divide by num of year in business
    df["bussiness_salaries"] = df["Net_Annual_Income"].astype("float")/(1+df["Years_At_Business"].values)

    ## change all status different to married to "Single"
    df["Marital_Status"]=np.where((df["Marital_Status"]!="Married"),"Alone",df["Marital_Status"])

    ## Income divide by Nb_Poject
    df["Nb_salaries"]=df["Net_Annual_Income"].astype("float")/(1+df["Nb_Of_Products"].values)

    ##dependance divide by num of year in business

    df["money_by_person"]=df["Net_Annual_Income"]/(1+df["Number_Of_Dependant"])

    ### Process for ratio_age_bussiness
    df['YEARS_BINNED'] = pd.cut(df['join_age'], bins = [0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
    age_data=df[["join_age","YEARS_BINNED","Y"]]
    
    #Group by the bin and calculate averages
    age_groups  = age_data.groupby('YEARS_BINNED').mean()
    age_groups = age_groups.rename(columns={"join_age": "mean_age"})
    age_groups = age_groups.drop("Y", axis=1)

    df = pd.merge(df,age_groups,on = 'YEARS_BINNED',how = 'left')
    ##mean_age divide by Years_At_Residence
    # df["ratio_age_business"]=df["mean_age"]/(1+df["Years_At_Residence"])
    df=df.drop(["join_age","YEARS_BINNED","mean_age"],axis=1)

    return reduce_mem_usage(df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get df path")
    parser.add_argument("--path", action="store", help="path to df")
    args = parser.parse_args()
    
    df = prepocess_data(args.path)
    df = reduce_mem_usage(df)
    print(df)