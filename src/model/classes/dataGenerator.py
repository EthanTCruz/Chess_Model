import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from Chess_Model.src.model.classes.game_analyzer import game_analyzer
import numpy as np
import chess
from Chess_Model.src.model.config.config import Settings
import random
from joblib import dump, load
import shutil
import csv
import math



class data_generator():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        


        if "epochs" not in kwargs:
            self.epochs=100
        else:
            self.epochs = kwargs["epochs"]

        if "target_feature" not in kwargs:
            self.target_feature = "w/b"
        else:
            self.target_feature = kwargs["target_feature"]
        
        if "predictions_board" not in kwargs:
            self.predictions_board='predictions.csv'
        else:
            self.predictions_board = kwargs["predictions_board"]

        if "batch_size" not in kwargs:
            self.batch_size = s.nnBatchSize
        else:
            self.batch_size = kwargs["batch_size"]

        
                
        if "scalerFile" not in kwargs:
            self.scalerFile = s.scaler_weights
        else:
            self.scalerFile = kwargs["scalerFile"]


        if "filename" not in kwargs:
            self.filename = "data.csv"
        else:
            self.filename = kwargs["filename"]


        if "test_size" not in kwargs:
            self.test_size=.2
        else:
            self.test_size=kwargs["test_size"]

        
        if "random_state" not in kwargs:
            self.random_state=42
        else:
            self.random_state = kwargs["random_state"]

        self.train_file = s.trainingFile
        self.test_file = s.testingFile
        self.gen_batch_size = s.nnGenBatchSize
        self.copy_data = s.copy_file
        self.validation_file = s.validationFile
        self.validation_size = s.nnValidationSize
        self.shape = (self.count_csv_headers(filename=self.filename) - 2)


    def get_shape(self):
        return self.shape
        
    def dataset_from_generator(self,filename,batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        dataset = tf.data.Dataset.from_generator(
            lambda: self.scaled_data_generator(filename=filename, batch_size=batch_size),
            output_types=(tf.float32, tf.float32),  # Update these types based on your data
            output_shapes=([None, self.shape], [None, 1])  # Update shapes based on your data
        )
        return dataset

    def create_scaler(self):
        scaler = StandardScaler()
        limit = self.get_row_count(self.train_file)
        batch_amt = 0
        for batch in self.data_generator(batch_size=self.gen_batch_size,filename=self.train_file):
            if batch_amt > (limit-1):
                break
            else:
                scaler.partial_fit(batch[0])
                batch_amt += self.gen_batch_size
        self.init_scaler(scaler=scaler)
        return scaler
    
    def init_scaler(self,scaler:StandardScaler):
        dump(scaler, self.scalerFile)
        return 0
    
    def initialize_datasets(self):
        self.copy_csv(source_file=self.filename, destination_file=self.copy_data)
        self.split_csv()
        self.create_scaler()

    def count_csv_headers(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Read the first row, which contains the headers
            return len(headers)
    
    def data_generator(self, batch_size,filename):
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                X, Y = self.clean_data(chunk)
                Y = np.array(Y)
                output = np.reshape(Y,(-1,1))
                yield (X, output)

    def load_scaler(self):
        scaler = load(self.scalerFile)
        return scaler
    
    def scaled_data_generator(self, batch_size,filename):
        scaler = self.load_scaler()
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                    X, Y = self.clean_data(chunk)
                    X_scaled = scaler.transform(X)
                    Y = np.array(Y)
                    output = np.reshape(Y,(-1,1))
                    yield (X_scaled, output)
        


    def clean_data(self,data):
        data = data.drop(columns=['moves(id)'])

        w_indices = data[data['w/b'] == 'w'].index
        b_indices = data[data['w/b'] == 'b'].index


        new_indices = np.concatenate([w_indices, b_indices])


        data = data.loc[new_indices]
        data[self.target_feature] = data[self.target_feature].apply(lambda x: 1 if x == 'w' else 0)


        # Split the data into features and target
        X = data.drop(columns=[self.target_feature])
        Y = data[self.target_feature]

        return X,Y
    

    def get_row_count(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)

    def test_dataset_size(self,filename = None):
        if filename is None:
            filename = self.test_file
        return self.get_row_count(filename=filename)

    def count_values_in_column(self,chunksize:int  = 10000):
        column_name = 'w/b'

        w_count = 0
        b_count = 0
        for chunk in pd.read_csv(self.filename, chunksize=chunksize, usecols=[column_name]):
            value_counts = chunk[column_name].value_counts()
            # Determine rows for train and test set
            w_count += value_counts.get('w')
            b_count += value_counts.get('b')
        limit = min(w_count,b_count)
        return limit


    def copy_csv(self,source_file, destination_file):
        shutil.copy(source_file, destination_file)



    def split_csv(self, chunksize=10000):
        self.copy_csv(source_file=self.filename, destination_file=self.copy_data)
        self.filename = self.copy_data
        total_rows = self.get_row_count(filename=self.filename)

        #make sure no shared inices
        # Split indices for training+testing and validation
        validation_size = self.validation_size  # 20% of the data for validation
        train_test_indices = set(range(total_rows))
        validation_indices = set(random.sample(list(train_test_indices), int(total_rows * validation_size)))

        train_test_indices -= validation_indices  # Remove validation indices from training+testing pool

        # Further split training+testing indices into training and testing
        test_indices = set(random.sample(list(train_test_indices), int(len(train_test_indices) * self.test_size)))

        processed_rows = 0
        processed_w_count = 0
        processed_b_count = 0
        max_wins = self.count_values_in_column(chunksize=chunksize)

        for chunk in pd.read_csv(self.filename, chunksize=chunksize):
            w_rows = chunk[chunk['w/b']=='w']
            b_rows = chunk[chunk['w/b']=='b']

            # Limit the rows based on processed count and min_count
            w_rows = w_rows.head(min(max_wins - processed_w_count, len(w_rows)))
            b_rows = b_rows.head(min(max_wins - processed_b_count, len(b_rows)))



            if processed_w_count >= max_wins:

                chunk_balanced = b_rows
            elif processed_w_count >= max_wins:

                chunk_balanced = w_rows
            else:
                chunk_balanced = pd.concat([w_rows, b_rows]).sample(frac=1)

            # Update processed counts
            processed_w_count += len(w_rows)
            processed_b_count += len(b_rows)


            chunk_train = chunk_balanced.iloc[[i - processed_rows in train_test_indices and i - processed_rows not in test_indices for i in range(processed_rows, processed_rows + len(chunk_balanced))]]
            chunk_test = chunk_balanced.iloc[[i - processed_rows in test_indices for i in range(processed_rows, processed_rows + len(chunk_balanced))]]
            chunk_validation = chunk_balanced.iloc[[i - processed_rows in validation_indices for i in range(processed_rows, processed_rows + len(chunk_balanced))]]

            # Write to respective files
            mode = 'a' if processed_rows > 0 else 'w'
            chunk_train.to_csv(self.train_file, mode=mode, index=False, header=(mode == 'w'))
            chunk_test.to_csv(self.test_file, mode=mode, index=False, header=(mode == 'w'))
            chunk_validation.to_csv(self.validation_file, mode=mode, index=False, header=(mode == 'w'))

            # Update processed rows counter
            processed_rows += len(chunk)







