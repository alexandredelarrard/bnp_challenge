from steps.step_data_load import DataLoader
from steps.step_data_prep import DataPreparation
from data_models.modelling_lightgbm import TrainModel, evaluation_r_auc

import numpy as np

if __name__ == "__main__":

    config_path = "./configs/main.yml"

    # data loading
    loader = DataLoader(config_path)
    datas= loader.run()

    # data prep
    prep = DataPreparation()
    train, test = prep.run(datas)

    # training 
    train_step = TrainModel(loader.config.modelling.config_lgbm, data=train)

    train["WEIGHT"] = np.where(train["FRAUD_FLAG"] == 1, 3, 1)
    total_test, model = train_step.modelling_cross_validation(data=train)
    test = train_step.test_on_set(model, test)
    evaluation_r_auc(total_test["FRAUD_FLAG"], total_test["PREDICTION_FRAUD_FLAG"])

    # 0.20648211314465909 -> 	0.1940

    submission = test[["ID", "PREDICTION_FRAUD_FLAG"]].rename(columns={"PREDICTION_FRAUD_FLAG" : "FRAUD_FLAG"}).reset_index()
    submission["FRAUD_FLAG"] = submission["FRAUD_FLAG"].clip(0,1)
    submission.columns = ["index", "ID", "fraud_flag"]
    submission.to_csv(loader.base_path + "/submission_300723_2.csv", sep=",", index=False)
    