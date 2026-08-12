import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass  #THIS IS A WRAPPER WHICH MAKES UP FOR THE MISSING __INIT__ IN THE CLASS BELOW
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')  #NOTE THAT HERE THE VARIABLES ARE NOT DIRECTLY EQUATED WITH THE PATH STRING, AS ON LINUX AND WINDOWS THE PATHS FOR SAME THING CAN EB DIFFERENT
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #THIS CREATES AN OBJECT OF THE CONFIG CLASS AND NOW THE PATHS OF THE STORING OF DATA IS CONTAINED IN THIS OBJECT

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\data.csv')
            logging.info('Read teh dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #THIS MAKES THE DIRECTORY WHERE THE DATA WILL BE STORED

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #THIS STORES THE WHOLE DATA AT THE DIRECTORY

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) #HERE THE FIRST PARAMETER IS THE PATH WHICH WILL BE CREATED

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
