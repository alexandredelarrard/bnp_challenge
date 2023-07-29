#
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils.general_functions import smart_column_parser

def remove_punctuation(x):
        x = x.str.replace(r'[^\w\s]+', '')
        x = x.str.replace(r' +', ' ')
        return x

def get_fraudulent_items(df, column):

    fraude = df.loc[df["FRAUD_FLAG"] == 1]
    f = fraude[[f"{column}1"]]

    for j in range(2,25):
            f = pd.concat([f, fraude[[f"{column}{j}"]].rename(columns={f"{column}{j}" : f"{column}1"})], axis=0)
    
    items = list(f[f"{column}1"].value_counts().loc[f[f"{column}1"].value_counts()>2].index)
    return items 

def concatenate_columns(df, cols):
    return df[[x for x in df.columns if cols in x and x != "NB_OF_ITEMS"]].apply(lambda x : list(x), axis=1)

def create_cols_freq(df, items, col):
    
    for item in list(items):
            df[item] = 0
    
    for item in list(items):
        for i in range(24):
            df[item] = df[item] + df[col].apply(lambda x: 1*(item == x[i]))*df[f"NBR_OF_PROD_PURCHAS{i+1}"].fillna(0)
    
    return df

def deduce_cash(df):
    df["CASH"] = concatenate_columns(df, cols="CASH_PRICE")
    df["CASH_MEAN"] = df["CASH"].apply(lambda x : np.nanmean(x))
    df["CASH_STD"] = df["CASH"].apply(lambda x : np.nanstd(x))
    df["CASH_MIN"] = df["CASH"].apply(lambda x : np.nanmin(x))
    df["CASH_MAX"] = df["CASH"].apply(lambda x : np.nanmax(x))
    df["CASH_TOTAL"] = df["CASH"].apply(lambda x : np.nansum(x))
    return df


class DataPreparation(object):

    def __init__(
        self,
    ):
        self.x = 0

    def run(self, datas):

        datas = datas.copy()
        
        # merge target to x train
        datas["x_train"] = datas["x_train"].merge(datas["y_train"][["ID", "fraud_flag"]], on="ID", how="left", validate="1:1")

        # load all datas
        x_train, items, makes = self.data_preparation_train(datas["x_train"])
        x_test = self.data_preparation_test(datas["x_test"], items, makes)

        return x_train, x_test
    
    def data_preparation_train(self, df):

        df.columns = smart_column_parser(df.columns)

        for col in df.columns:
             if df[col].dtype == "O":
                df[col] = remove_punctuation(df[col])

        # items counting 
        df["ITEMS"] = concatenate_columns(df, cols="ITEM")
        items = get_fraudulent_items(df, column="ITEM")
        df = create_cols_freq(df, items, col="ITEMS")

        # brand counting 
        df["MAKES"] = concatenate_columns(df, cols="MAKE")
        makes = get_fraudulent_items(df, column="MAKE")
        df = create_cols_freq(df, makes, col="MAKES")

        # prices 
        df = deduce_cash(df)

        # models 
        df["MODEL1"] = df["MODEL1"].astype("category")
        df["MACBOOK"] = 1*df[["MODEL1", "MODEL2", "MODEL3"]].apply(lambda x : "APPLE MACBOOK PRO" in str(x[0]) + str(x[1]) + str(x[2]) or "APPLE MACBOOK AIR" in str(x[0]) + str(x[1]) + str(x[2]), axis=1)
        df["IPAD"] = 1*df[["MODEL1", "MODEL2", "MODEL3"]].apply(lambda x : "APPLE IPAD PRO" in str(x[0]) + str(x[1]) + str(x[2])  or "APPLE IPAD AIR" in str(x[0]) + str(x[1]) + str(x[2]), axis=1)
        df["GALAXY"] = 1*df[["MODEL1", "MODEL2", "MODEL3"]].apply(lambda x : "SAMSUNG GALAXY" in str(x[0]) + str(x[1]) + str(x[2]), axis=1)
        df["WATCH"] = 1*df[["MODEL1", "MODEL2", "MODEL3"]].apply(lambda x : "APPLE WATCH" in str(x[0]) + str(x[1]) + str(x[2]), axis=1)

        #total bought 
        df["NBR_OF_PROD_PURCHAS"] = df[[x for x in df.columns if "NBR_OF_PROD_PURCHAS" in x]].fillna(0).sum(axis=1)
        df["NBR_OF_PROD_PURCHAS"] = df["NBR_OF_PROD_PURCHAS"].clip(0,20)

        # retailer code 
        df["GOODSCODE"] = np.where(df["GOODS_CODE2"] == "FULFILMENT", "FULFILMENT",
                          np.where(df["GOODS_CODE2"].apply(lambda x : "DMS" in str(x)), "DMS",
                          np.where(df["GOODS_CODE2"].isnull(), "MISSING", "RETAILER_ID")))

        return df, items, makes
    
    def data_preparation_test(self, df, items, makes):
         
        df.columns = smart_column_parser(df.columns)

        for col in df.columns:
             if df[col].dtype == "O":
                df[col] = remove_punctuation(df[col])

        # items counting 
        df["ITEMS"] = concatenate_columns(df, cols="ITEM")
        df = create_cols_freq(df, items, col="ITEMS")

        # brand counting 
        df["MAKES"] = concatenate_columns(df, cols="MAKE")
        df = create_cols_freq(df, makes, col="MAKES")

        # prices 
        df = deduce_cash(df)

        # models 
        df["MODEL1"] = df["MODEL1"].astype("category")
        df["MACBOOK"] = 1*df[["MODEL1", "MODEL2", "MODEL3"]].apply(lambda x : "APPLE MACBOOK PRO" in str(x[0]) + str(x[1]) + str(x[2]) or "APPLE MACBOOK AIR" in str(x[0]) + str(x[1]) + str(x[2]), axis=1)
        df["IPAD"] = 1*df[["MODEL1", "MODEL2", "MODEL3"]].apply(lambda x : "APPLE IPAD PRO" in str(x[0]) + str(x[1]) + str(x[2])  or "APPLE IPAD AIR" in str(x[0]) + str(x[1]) + str(x[2]), axis=1)
        df["GALAXY"] = 1*df[["MODEL1", "MODEL2", "MODEL3"]].apply(lambda x : "SAMSUNG GALAXY" in str(x[0]) + str(x[1]) + str(x[2]), axis=1)
        df["WATCH"] = 1*df[["MODEL1", "MODEL2", "MODEL3"]].apply(lambda x : "APPLE WATCH" in str(x[0]) + str(x[1]) + str(x[2]), axis=1)


        #total bought 
        df["NBR_OF_PROD_PURCHAS"] =df[[x for x in df.columns if "NBR_OF_PROD_PURCHAS" in x]].fillna(0).sum(axis=1)
        df["NBR_OF_PROD_PURCHAS"] = df["NBR_OF_PROD_PURCHAS"].clip(0,20)

        # retailer code 
        df["GOODSCODE"] = np.where(df["GOODS_CODE2"] == "FULFILMENT", "FULFILMENT",
                          np.where(df["GOODS_CODE2"].apply(lambda x : "DMS" in str(x)), "DMS",
                          np.where(df["GOODS_CODE2"].isnull(), "MISSING", "RETAILER_ID")))

        return df



        