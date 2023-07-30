#
# -*- coding: utf-8 -*-

import pandas as pd
import string
import re
import numpy as np
from utils.general_functions import smart_column_parser


def remove_punctuation(input_string):
    # Make a translator object to replace punctuation with none
    translator = str.maketrans('', '', string.punctuation)
    # Use the translator
    input_string = input_string.translate(translator)
    input_string = re.sub(" +", " ", input_string)
    return input_string

def clean_text(x):
    x = x.apply(lambda x : remove_punctuation(str(x)).strip())
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
        df[item] = df[item].clip(0,3)

    return df

def group_items_category(df):
    items = df["ITEM1"].unique()

    mapping = {}
    for v in items:
        mapping[v] = v

    mapping["BABY PLAY EQUIPMENT"] = "BABY EQUIPMENT"
    mapping["BABY FEEDING"] = "BABY EQUIPMENT"
    mapping["NURSERY LINEN"] = "BABY EQUIPMENT"
    mapping["NURSERY FURNITURE"] = "BABY EQUIPMENT"
    mapping["BABYWEAR"] = "BABY EQUIPMENT"
    mapping["NURSERY ACCESSORIES"] = "BABY EQUIPMENT"
    mapping["BABY CHANGING"] = "BABY EQUIPMENT"
    mapping["BABY CHILD TRAVEL"] = "BABY EQUIPMENT"

    mapping["BAGS CARRY CASES"] = "LUGGAGE"
    mapping["MEN S FOOTWEAR"] = "MEN S CLOTHES"
    mapping["TELEPHONE ACCESSORIES"] = "AUDIO ACCESSORIES"

    return mapping

def models(df):

    df["MODEL"] = df[[x for x in df.columns if "MODEL" in x]].apply(lambda x : " ".join(x), axis=1)

    df["GALAXY"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('(SAMSUNG GALAXY).*(S7|S21|S22|Z FOLD).*$', x))>0 else 0)
    df["GALAXY_WATCH"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('(SAMSUNG GALAXY).*(WATCH 4).*$', x))>0 else 0)
    
    df["SAMSUNG_QLED"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('(SAMSUNG ).*(2021|2020).*(QLED|NEO QLED).*$', x))>0 else 0)
    df["AIRPODS"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('^.*(APPLE AIRPODS|AIRPODS).*(MAX|PRO).*$', x))>0 else 0)
    
    df["IPAD"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('^.*(2020|2021)( APPLE IPAD )(AIR|PRO).*$', x))>0 else 0)
    df["IPHONE"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('^.*(APPLE IPHONE ).*(11|12|13).*$', x))>0 else 0)
    df["WATCH"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('^.*(APPLE WATCH).*(SERIES).*(6|7).*$', x))>0 else 0)
    df["MACBOOK"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('^.*(2019|2020|2021).*(APPLE MACBOOK|APPLE MAC MINI).*(PRO|AIR).*$', x))>0 else 0)
    
    df["LG_OLED"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('^.*(LG OLED|LG).*(2021 LED|2021 OLED|2020 OLED).*$', x))>0 else 0)
    df["SONY_OLED"] = df["MODEL"].apply(lambda x : 1 if len(re.findall('^.*(SONY).*(2021 LED|2021 OLED|2020 OLED).*$', x))>0 else 0)

    df["TOTAL_APPLE"] = df["IPAD"] +df["IPHONE"]+df["WATCH"]+df["MACBOOK"]+df["AIRPODS"]
    df["TOTAL_APPLE"] = df["TOTAL_APPLE"].clip(0,2)

    return df


def deduce_cash(df):
    df["CASH"] = concatenate_columns(df, cols="CASH_PRICE")
    df["CASH_MEAN"] = df["CASH"].apply(lambda x : np.nanmean(x))
    df["CASH_STD"] = df["CASH"].apply(lambda x : np.nanstd(x))
    df["CASH_MIN"] = df["CASH"].apply(lambda x : np.nanmin(x))
    df["CASH_MAX"] = df["CASH"].apply(lambda x : np.nanmax(x))
    df["CASH_TOTAL"] = df["CASH"].apply(lambda x : np.nansum(x))
    return df

def sum_items(df):

    df["ELECTRONICS"] = (df['COMPUTERS'] + 
                         df['TELEPHONES FAX MACHINES TWOWAY RADIOS']+
                         df['COMPUTER PERIPHERALS ACCESSORIES']
                         ).clip(0,6)
    
    df["AUDIO"] = (df['TELEVISIONS HOME CINEMA']+
                    df['AUDIO ACCESSORIES']+
                    df['HIFI']+
                    df['IMAGING EQUIPMENT']+
                    df['POWER BATTERIES']
                ).clip(0,6)

    df["CLOTHS"] = (df['BATH LINEN'] + 
                    df['MEN S CLOTHES']+
                    df['BABY EQUIPMENT']+
                    df['LAUNDRY CLOTHESCARE'] +
                    df['LUGGAGE']
                    ).clip(0,6)    
    
    df["HOME"] = ( + 
                df['TABLEWARE']+
                df['BED LINEN']+
                df['DECORATIVE ACCESSORIES']+
                df['LIGHTING']+
                df["SOFT FURNISHINGS"]+
                df["KITCHEN UTENSILS GADGETS"]+
                df["STORAGE ORGANISATION"] +
                df["BATHROOM ACCESSORIES"]+
                df["COOKWARE"]+
                df["LIVING DINING FURNITURE"] +
                df["HOUSEHOLD CLEANING"]
                ).clip(0,6)    

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
                df[col] = clean_text(df[col])

        mapping = group_items_category(df)

        for col in [x for x in df.columns if "ITEM" in x and x != "NB_OF_ITEMS"]:
             df[col] = df[col].map(mapping)

        # items counting 
        df["ITEMS"] = concatenate_columns(df, cols="ITEM")
        items = get_fraudulent_items(df, column="ITEM")
        df = create_cols_freq(df, items, col="ITEMS")
        df = sum_items(df)

        # brand counting 
        df["MAKES"] = concatenate_columns(df, cols="MAKE")
        makes = get_fraudulent_items(df, column="MAKE")
        df = create_cols_freq(df, makes, col="MAKES")

        # prices 
        df = deduce_cash(df)

        # models 
        df = models(df)
        
        #total bought 
        df["NBR_OF_PROD_PURCHAS"] = df[[x for x in df.columns if "NBR_OF_PROD_PURCHAS" in x]].fillna(0).sum(axis=1)
        df["NBR_OF_PROD_PURCHAS"] = df["NBR_OF_PROD_PURCHAS"].clip(0,15)

        # retailer code 
        df["GOODSCODE"] = df[[x for x in df.columns if "GOODS_CODE" in x]].apply(lambda x : " ".join(x), axis=1)
        df["DMS"] = df["GOODSCODE"].apply(lambda x : "DMS" in x)
        df["FULFILMENT"] = df["GOODSCODE"].apply(lambda x : "FULFILMENT" in x)

        df["CODE_1_RETAILER"] = df["GOODS_CODE1"].astype(str).str[:3].astype("category")
        df["CODE_1_RETAILER"] = np.where(df["CODE_1_RETAILER"]=="SKU", "140", df["CODE_1_RETAILER"]).astype(int)

        return df, items, makes
    
    def data_preparation_test(self, df, items, makes):
         
        df.columns = smart_column_parser(df.columns)
        df["GOODS_CODE23"] = df["GOODS_CODE23"].astype(str)
        df["GOODS_CODE24"] = df["GOODS_CODE24"].astype(str)

        for col in df.columns:
            if df[col].dtype == "O":
                df[col] = clean_text(df[col])

        mapping = group_items_category(df)

        for col in [x for x in df.columns if "ITEM" in x and x != "NB_OF_ITEMS"]:
             df[col] = df[col].map(mapping)

        # items counting 
        df["ITEMS"] = concatenate_columns(df, cols="ITEM")
        df = create_cols_freq(df, items, col="ITEMS")
        df = sum_items(df)

        # brand counting 
        df["MAKES"] = concatenate_columns(df, cols="MAKE")
        df = create_cols_freq(df, makes, col="MAKES")

        # prices 
        df = deduce_cash(df)

        # models 
        df = models(df)

        #total bought 
        df["NBR_OF_PROD_PURCHAS"] =df[[x for x in df.columns if "NBR_OF_PROD_PURCHAS" in x]].fillna(0).sum(axis=1)
        df["NBR_OF_PROD_PURCHAS"] = df["NBR_OF_PROD_PURCHAS"].clip(0,15)

        # retailer code 
        df["GOODSCODE"] = df[[x for x in df.columns if "GOODS_CODE" in x]].apply(lambda x : " ".join(x), axis=1)
        df["DMS"] = df["GOODSCODE"].apply(lambda x : "DMS" in x)
        df["FULFILMENT"] = df["GOODSCODE"].apply(lambda x : "FULFILMENT" in x)

        df["CODE_1_RETAILER"] = df["GOODS_CODE1"].astype(str).str[:3]
        df["CODE_1_RETAILER"] = np.where(df["CODE_1_RETAILER"]=="SKU", "140", df["CODE_1_RETAILER"]).astype(int)

        return df



        