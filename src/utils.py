import os,sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from src.logger import logging
from src.exception import CustomException
import dill,json
def save_model(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            logging.info(f"Evaluation initiated for {model}.")
            para = param[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=10,verbose=1,n_jobs=-1)
            logging.info(f"GridSearchCV initiated for {model}.")
            gs.fit(x_train,y_train)
            logging.info(f"GridSearchCV fit done and set_params initiated for {model}.")
            model.set_params(**gs.best_params_)
            logging.info(f"setting parameters completed and fitting initiated for {model}.")
            model.fit(x_train,y_train)
            logging.info(f"prediction initiated for {model}.")
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            logging.info(f"Getting the accuracy for train and test data for {model}")
            train_model_accuracy = accuracy_score(y_true=y_train,y_pred=y_train_pred)
            test_model_accuracy = accuracy_score(y_true=y_test,y_pred=y_test_pred)
            report[list(models.keys())[i]] = test_model_accuracy
            logging.info(f"Obtained accuracy of {test_model_accuracy} and completed with {model}.")
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def save_json_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'w') as f:
            json.dump(obj,f)
    except Exception as e:
        raise CustomException(e,sys)