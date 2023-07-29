#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import pandas as pd
from steps.step import Step

class DataLoader(Step):

    def __init__(
        self,
        config_path: str
    ):
        super().__init__(config_path=config_path)
        self.base_path = self.config.load.base["data_path"]

    def run(self):
        
        # load all datas
        datas = self.data_loading()

        return datas
    
    def data_loading(self):

        datas = {}

        datas["x_train"] = pd.read_csv(self.base_path + "/X_train.csv")
        datas["y_train"] = pd.read_csv(self.base_path + "/Y_train.csv")
        datas["x_test"] = pd.read_csv(self.base_path + "/X_test.csv")
    
        return datas

        