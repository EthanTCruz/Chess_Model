import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from Chess_Model.src.model.config.config import Settings
from Chess_Model.src.model.classes.cnn_game_analyzer import game_analyzer

import random
from joblib import dump, load, Parallel, delayed
import shutil
import csv
import math
import re
import os
import time

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

        if "matrixScalerFile" not in kwargs:
            self.matrixScalerFile = s.matrixScalerFile
        else:
            self.matrixScalerFile = kwargs["matrixScalerFile"]

        if "filename" not in kwargs:
            self.filename = s.scores_file
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
        self.scalarBatchSize = s.nnScalarBatchSize

        self.recordsDataFile = s.recordsData

        self.recordsDataFileCopy = s.recordsDataCopy
        self.recordsDataFileTrain = s.recordsDataTrain
        self.recordsDataFileTest = s.recordsDataTest
        self.recordsDataFileValidation = s.recordsDataValidation

        evaluator = game_analyzer(scores_file=s.scores_file)
        evaluator.set_feature_description()
        self.feature_description = evaluator.get_feature_description()

        self.seed=3141





        



    
    def init_scaler(self,scaler:StandardScaler,scalarFile: str = None):
        if scalarFile is None:
            scalarFile = self.scalerFile
        dump(scaler, scalarFile)
        self.scalar = scaler
        return 0
    

    def load_scaler(self,scalarFile: str = None):
        if scalarFile is None:
            scalarFile = self.scalerFile
        self.scaler = load(scalarFile)
        return self.scaler
 
    

    def get_row_count(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)

    def test_dataset_size(self,filename = None):
        if filename is None:
            filename = self.test_file
        return self.get_row_count(filename=filename)


    def copy_csv(self,source_file, destination_file):
        shutil.copy(source_file, destination_file)

    def preprocess_and_scale_chunk(self, chunk):
        # Preprocess the chunk. Adjust this to your actual preprocessing needs
        X, _, _ = self.clean_data(chunk)
        # Assuming X is now a clean, preprocessed DataFrame
        return X

    def create_scaler_cpu(self):

        
        scaler = StandardScaler()
        chunks = pd.read_csv(self.train_file, chunksize=100)
        
        # Preprocess chunks in parallel
        preprocessed_chunks = Parallel(n_jobs=-1)(
            delayed(self.preprocess_and_scale_chunk)(chunk) for chunk in chunks
        )
        
        # Combine all preprocessed chunks into one DataFrame if possible or fit scaler incrementally
        for preprocessed_chunk in preprocessed_chunks:
            scaler.partial_fit(preprocessed_chunk)
        
        self.init_scaler(scaler)

        return scaler


    def initialize_datasets(self):
        self.split_csv()
        self.headers = pd.read_csv(self.train_file,nrows=0)
        self.non_matrix_headers = [col for col in self.headers.columns if not col.endswith('positions')]
        self.create_scaler()
        #self.create_scaler_cpu()
        self.shape = self.get_shape()

    def split_csv(self, chunksize=10000):
        if os.path.exists(self.train_file):
            os.remove(self.train_file)
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.validation_file):
            os.remove(self.validation_file)
        if os.path.exists(self.copy_data):
            os.remove(self.copy_data)
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


        for chunk in pd.read_csv(self.filename, chunksize=chunksize):


            chunk_train = chunk.iloc[[i - processed_rows in train_test_indices and i - processed_rows not in test_indices for i in range(processed_rows, processed_rows + len(chunk))]]
            chunk_test = chunk.iloc[[i - processed_rows in test_indices for i in range(processed_rows, processed_rows + len(chunk))]]
            chunk_validation = chunk.iloc[[i - processed_rows in validation_indices for i in range(processed_rows, processed_rows + len(chunk))]]

            # Write to respective files
            mode = 'a' if processed_rows > 0 else 'w'
            chunk_train.to_csv(self.train_file, mode=mode, index=False, header=(mode == 'w'))
            chunk_test.to_csv(self.test_file, mode=mode, index=False, header=(mode == 'w'))
            chunk_validation.to_csv(self.validation_file, mode=mode, index=False, header=(mode == 'w'))

            # Update processed rows counter
            processed_rows += len(chunk)

    def clean_data(self, data):
        # Create DataFrame for columns with 'positions' suffix
        if data is None:
            return None,None,None
        matrix_data = pd.DataFrame()
        for col in self.matrix_headers:
            matrix_data[col] = data[col].apply(lambda x: flat_string_to_array(x))

        # DataFrame without 'positions' columns


        # Split the non-positions data into features and target
        X = data[self.non_matrix_headers]
        X = X.drop(columns=self.target_feature)
        Y = data[self.target_feature]

        return X, Y, matrix_data
    
    def create_scaler(self):
        scaler = StandardScaler()
        
        limit = self.get_row_count(self.train_file)
        adjusted_limit = math.ceil(limit/self.scalarBatchSize)
        batch_amt = 0
        batches = self.data_generator_no_matrices(batch_size=self.scalarBatchSize,filename=self.train_file)
        for batch in  tqdm(batches, total=adjusted_limit, desc="Creating Scalar"):
            if batch_amt > (limit-1):
                break
            else:

                scaler.partial_fit(batch[1])
                batch_amt += self.scalarBatchSize
        self.init_scaler(scaler=scaler)

        return scaler
    
    def data_generator(self, batch_size,filename):
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                X, Y, matrix_data = self.clean_data(chunk)
                Y = np.array(Y)
                #output = np.reshape(Y,(-1,1))
                output = Y
                yield (matrix_data, X, output )

    def data_generator_no_matrices(self, batch_size,filename):
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size,usecols=self.non_matrix_headers)
            for chunk in data:
                X, Y, matrix_data = self.clean_data(chunk)
                Y = np.array(Y)

                output = Y
                yield (matrix_data, X, output )
                
    def scaled_data_generator(self, batch_size,filename):


        while True:
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                    X, Y, matrixData = self.clean_data(chunk)
                    X_scaled = self.scaler.transform(X)
                    #print(matrixData.columns)
                    matrixData_scaled = matrixData.stack().map(reshape_to_matrix).unstack()

                    output_matrices = np.stack(matrixData_scaled.apply(lambda row: np.stack(row, axis=-1), axis=1).to_numpy())

                    Y = np.array(Y)

                    output = Y
                    yield ((output_matrices,X_scaled), output)


    def get_shape(self):
        self.headers = pd.read_csv(self.train_file,nrows=0)
        self.non_matrix_headers = [col for col in self.headers.columns if not col.endswith('positions')]
        self.matrix_headers = [col for col in self.headers.columns if col.endswith('positions')]
        matrix_channels = 0
        metadata_columns = 0
        batch = next(self.data_generator(batch_size=self.gen_batch_size,filename=self.train_file))
        matrix_channels = len(batch[0].columns)
        metadata_columns = len(batch[1].columns)
        matrix_shape = (8,8,matrix_channels)
        metadata_shape = (metadata_columns,)
        self.shape = [matrix_shape,metadata_shape]

        self.train_data = self.dataset_from_generator(filename=self.train_file)
        self.test_data = self.dataset_from_generator(filename=self.test_file)
        self.validation_data = self.dataset_from_generator(filename=self.validation_file)
        self.load_scaler()


        return self.shape
    
    def dataset_from_generator(self,filename,batch_size: int = None):
        if batch_size is None:
            batch_size = self.gen_batch_size
        shapes = self.shape
        matrix_shape = shapes[0]
        meta_shape = shapes[1]
        dataset = tf.data.Dataset.from_generator(
            lambda: self.scaled_data_generator(filename=filename, batch_size=batch_size),
            output_types=((tf.int8, tf.float16),tf.float16),  
            output_shapes=(([self.gen_batch_size, matrix_shape[0], matrix_shape[1], matrix_shape[2]], 
                            [self.gen_batch_size, meta_shape[0]]), 
                            [self.gen_batch_size, 3]) 
        )
        return dataset
    
    def _parse_function(self,example_proto):
        # Parse the input tf.train.Example proto using the feature description dictionary
        parsed_features = tf.io.parse_single_example(example_proto, self.feature_description)
        
        # Initialize dictionaries for the three categories
        positions_data = {}
        mean_data = {}
        other_data = {}
        
        # Decode features based on type and categorize
        for key, feature in self.feature_description.items():
            if feature.dtype == tf.string and key.endswith('positions'):
                # Parse, reshape to 8x8 for position/matrix data
                parsed_array = tf.io.parse_tensor(parsed_features[key], out_type=tf.int32)
                positions_data[key] = tf.reshape(parsed_array, [8, 8])
            elif key in ['white mean', 'black mean', 'stalemate mean']:
                # Directly assign mean data without need for decoding
                mean_data[key] = parsed_features[key]
            else:
                # Handle other data types, decode if necessary
                if feature.dtype == tf.string:
                    parsed_array = tf.io.parse_tensor(parsed_features[key], out_type=tf.int32)
                    other_data[key] = parsed_array
                else:
                    other_data[key] = parsed_features[key]
        
        return positions_data, other_data, mean_data
    
    def parser(self,recordsFile):
        tfrecord_filenames = [recordsFile]
        dataset = tf.data.TFRecordDataset(tfrecord_filenames)

        # Map the parsing function over the dataset
        parsed_dataset = dataset.map(self._parse_function)

        # # Iterate over the parsed dataset and use the data
        # for positions, others, means in parsed_dataset.take(5):
        #     print("Positions Data:", positions)
        #     print("Other Data:", others)
        #     print("Means Data:", means)
        #     print("\n---\n")
        return parsed_dataset

    def data_generator_no_matrices_records(self):
        for positions, others, means in self.parsed_dataset:
            yield others
    
    def get_row_count_records(self):
        # Initialize a counter
        count = 0

        for _ in self.parsed_dataset:
            count += 1
        return count

    def get_row_count_records_old(self):
        # Assuming the dataset is batched
        return sum(1 for _ in self.parsed_dataset.unbatch())

    def create_scaler_records(self):
        scaler = StandardScaler()
        
        # Adjusted for TensorFlow's dataset API
        total_rows = count_dataset_elements(dataset=self.parsed_dataset)
        adjusted_limit = math.ceil(total_rows / self.scalarBatchSize)
        
        # Convert dataset to generator for partial fitting
        batches = self.data_generator_no_matrices_records()
        
        # Loop through the dataset and partial_fit the scaler
        for _ in range(adjusted_limit):
            try:
                batch = next(batches)
                batch_values = [value.numpy() for value in batch.values()]
                batch_array = np.array(batch_values).reshape(1, -1) 
                scaler.partial_fit(batch_array)
            except StopIteration:
                break  # When dataset ends

        self.init_scaler(scaler=scaler)
        return scaler

    def initialize_datasets_records(self):
        self._split_tfrecord()
        self.parsed_dataset = self.parser(recordsFile=self.recordsDataFileTrain)
        self.create_scaler_records()
        # #self.create_scaler_cpu()
        # self.shape = self.get_shape()

    def _split_tfrecord(self):
        # random.seed(3141)
        # Load the dataset

        if os.path.exists(self.recordsDataFileTrain):
            os.remove(self.recordsDataFileTrain)
        if os.path.exists(self.recordsDataFileTest):
            os.remove(self.recordsDataFileTest)
        if os.path.exists(self.recordsDataFileValidation):
            os.remove(self.recordsDataFileValidation)
        if os.path.exists(self.recordsDataFileCopy):
            os.remove(self.recordsDataFileCopy)

        self.copy_csv(self.recordsDataFile,self.recordsDataFileCopy)
        self.recordsDataFile = self.recordsDataFileCopy

        raw_dataset = tf.data.TFRecordDataset(self.recordsDataFile)

        # Define split ratios
        train_ratio = 1.0 - (self.test_size + self.validation_size)
        
        # Probabilistically filter the dataset into train, validation, and test sets
        def is_train(x):
            return tf.random.uniform([], seed=self.seed) < train_ratio

        def is_val(x):
            return tf.logical_and(tf.random.uniform([], seed=self.seed) >= train_ratio, 
                                  tf.random.uniform([], seed=self.seed) < train_ratio + self.validation_size)

        def is_test(x):
            return tf.random.uniform([], seed=self.seed) >= (train_ratio + self.validation_size)

        train_dataset = raw_dataset.filter(is_train)
        val_dataset = raw_dataset.filter(is_val)
        test_dataset = raw_dataset.filter(is_test)

        # Function to write the split datasets to new TFRecord files
        def write_tfrecord(dataset, filename):
            tf.data.experimental.TFRecordWriter(filename).write(dataset)
        
        # Write the splits to new TFRecord files
        write_tfrecord(train_dataset, self.recordsDataFileTrain)
        write_tfrecord(val_dataset, self.recordsDataFileValidation)
        write_tfrecord(test_dataset, self.recordsDataFileTest)
        
        # train_size = count_dataset_elements(train_dataset)
        # val_size = count_dataset_elements(val_dataset)
        # test_size = count_dataset_elements(test_dataset)
        # total_size = train_size + test_size + val_size
        # print(f"Training set size: {train_size}, ratio = {train_size/total_size}")
        # print(f"Validation set size: {val_size}, ratio = {val_size/total_size}")
        # print(f"Testing set size: {test_size}, ratio = {test_size/total_size}")


                        
def string_to_array(s):
    # Safely evaluate the string as a Python literal (list of lists in this case)
    return np.array(ast.literal_eval(s))

def count_dataset_elements(dataset):
    # Attempt to use the cardinality if available
    cardinality = tf.data.experimental.cardinality(dataset).numpy()
    if cardinality in [tf.data.experimental.INFINITE_CARDINALITY, tf.data.experimental.UNKNOWN_CARDINALITY]:
        # Fallback: Iterate through the dataset and count elements
        return sum(1 for _ in dataset)
    else:
        return cardinality

def flat_string_to_array(s):
    if not s:
        return None

    # Remove all square brackets and replace newline characters with spaces
    clean_string = s.replace('[', '').replace(']', '').replace('\n', ' ')

    # Split the cleaned string on spaces to isolate the numbers as strings
    numbers_str = clean_string.split()

    # Convert list of number strings to a NumPy array of type int8
    try:
        integer_array = np.array(numbers_str, dtype=np.int8)
    except ValueError as e:
        # Handle cases where conversion fails due to invalid numeric strings
        print(f"Error converting string to array: {e}")
        return None

    return integer_array



def reshape_to_matrix(cell):
    return np.array(cell).reshape(8, 8)