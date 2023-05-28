import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from game_analyzer import game_analyzer
import numpy as np
from redis_populator import populator
import redis
import os

class neural_net():

    def __init__(self,filename,ModelFilePath="./",ModelFilename="chess_model",target_feature = "w/b",test_size=.2,random_state=42,player = "NA",predictions_board='predictions.csv',epochs=100) -> None:
        self.epochs = epochs
        self.target_feature = target_feature
        self.predictions_board = predictions_board
        self.filename = filename
        self.test_size=test_size
        self.random_state = random_state
        self.ModelFile = f"{ModelFilePath}{ModelFilename}"
        self.game_analyzer_obj = game_analyzer(output_file=self.filename,player=player)




    def clean_data(self,data):
        # Remove the moves column, as it's not a useful feature for the neural network
        data = data.drop(columns=['moves(id)'])

        # Encode the target variable (w/b) as 0 or 1
        data[self.target_feature] = data[self.target_feature].apply(lambda x: 1 if x == 'w' else 0)
        # One-hot encode the 'game time' feature

        # Split the data into features and target
        X = data.drop(columns=[self.target_feature])
        Y = data[self.target_feature]

        return X,Y

    def clean_prediction_data(self,data):
        # Remove the moves column, as it's not a useful feature for the neural network
        Y = data['moves(id)']
        data = data.drop(columns=['moves(id)'])

        # Encode the target variable (w/b) as 0 or 1
        data[self.target_feature] = data[self.target_feature].apply(lambda x: 1 if x == 'w' else 0)
        # One-hot encode the 'game time' feature

        # Split the data into features and target
        X = data.drop(columns=[self.target_feature])


        return X,Y




    def partition_and_scale(self,X,Y):
        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)

        # Scale the feature data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, Y_train, Y_test

    
    def create_and_evaluate_model(self):

        data = pd.read_csv(self.filename)
        X,Y = self.clean_data(data=data)
        
        X_train, X_test, Y_train, Y_test = self.partition_and_scale(X=X,Y=Y)
        # Create a simple neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, Y_train, batch_size=32, epochs=self.epochs, validation_split=0.2)
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test, Y_test)

        tf.keras.models.save_model(model=model,filepath=self.ModelFile)

        print("Test loss:", loss)
        print("Test accuracy:", accuracy)
        return model
    
    def process_redis_boards(self):        
        self.game_analyzer_obj.process_redis_boards()
        data = pd.read_csv(self.filename)

        X,moves = self.clean_prediction_data(data=data)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = tf.keras.models.load_model(filepath=self.ModelFile)

        predictions = model.predict(X)
        moves['predictions'] = predictions
        return(moves)

    def score_board(self,board_key):
        self.game_analyzer_obj.process_single_board(board_key=board_key)
        
        data = pd.read_csv(self.filename)
        X,Y = self.clean_data(data=data)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = tf.keras.models.load_model(filepath=self.ModelFile)

        prediction = float(model.predict(X))

        return(prediction)        




