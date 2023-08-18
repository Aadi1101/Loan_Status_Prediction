import os,sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.utils import save_model

@dataclass
class DataTransformationConfig():
    preprocessor_file_path = os.path.join('src/models','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,trainset_path,testset_path):
        try:
            train_df = pd.read_csv(trainset_path)
            test_df = pd.read_csv(testset_path)

            logging.info("Reading the train and test data completed now transforming null values of columns")
            
            train_df['Gender'].fillna(value='Female',inplace=True)
            test_df['Gender'].fillna(value='Female',inplace=True)
            
            train_df['Married'].fillna(value='No',inplace=True)
            test_df['Married'].fillna(value='No',inplace=True)

            train_df['Credit_History'].fillna(value=0.0,inplace=True)
            test_df['Credit_History'].fillna(value=0.0,inplace=True)

            train_df['Self_Employed'].fillna(value='Yes',inplace=True)
            test_df['Self_Employed'].fillna(value='Yes',inplace=True)

            logging.info("Dropping the null values which will not cause any effect to the prediction.")

            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)

            logging.info("Transforming the categorical columns to numeric.")

            train_df.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)
            test_df.replace({'Loan_Status':{'N':0,'Y':1}},inplace=True)

            train_df = train_df.replace(to_replace='3+',value=4)
            test_df = test_df.replace(to_replace='3+',value=4)

            train_df.replace({'Married':{'Yes':1,'No':0},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
            test_df.replace({'Married':{'Yes':1,'No':0},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

            logging.info("Obtaining the preprocessor object.")

            preprocessing_obj = StandardScaler()
            target_column_name = "Loan_Status"

            logging.info("Splitting the features and target column.")

            input_feature_train_df = train_df.drop(columns=[target_column_name,'Loan_ID'],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name,'Loan_ID'],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on both training and testing object.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info("Data Transformation complete now saving the preprocessing object")
            save_model(self.data_transformation_config.preprocessor_file_path,preprocessing_obj)
            return(
                train_arr,test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)