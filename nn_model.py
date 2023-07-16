import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from game_analyzer import game_analyzer
import numpy as np
from redis_populator import populator
import redis
import os
import chess


class neural_net():

    def __init__(self,filename,ModelFilePath="./",ModelFilename="chess_model",target_feature = "w/b",test_size=.2,random_state=42,player = "NA",predictions_board='predictions.csv',epochs=100,redis_client=redis.Redis(host='localhost', port=6379)) -> None:
        self.epochs = epochs
        self.target_feature = target_feature
        self.predictions_board = predictions_board
        self.r = redis_client
        self.filename = filename
        self.test_size=test_size
        self.random_state = random_state
        self.ModelFile = f"{ModelFilePath}{ModelFilename}"
        self.game_analyzer_obj = game_analyzer(output_file=self.filename,player=player)




    def clean_data(self,data):
        # Remove the moves column, as it's not a useful feature for the neural network
        data = data.drop(columns=['moves(id)'])
        b_count = data[data['w/b'] == 'b'].shape[0]
        w_count = data[data['w/b'] == 'w'].shape[0]

        # calculate number of 'w' rows to keep (which is 3/4 of 'w' count or 'b' count whichever is lower)
        w_keep = min(int(w_count * 0.75), b_count)

        # get indices of 'w' rows
        w_indices = data[data['w/b'] == 'w'].index

        # choose random subset of 'w' indices to keep
        w_indices_keep = np.random.choice(w_indices, w_keep, replace=False)

        # get all 'b' indices
        b_indices = data[data['w/b'] == 'b'].index

        # combine 'w' indices to keep and all 'b' indices
        new_indices = np.concatenate([w_indices_keep, b_indices])

        # filter dataframe to these indices
        data = data.loc[new_indices]
                
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

    def get_legal_moves(self,board: chess.Board):
        legal_moves =  [move.uci() for move in board.legal_moves]
        return(legal_moves)
        

    def evaluate_redis(self,board: chess.Board):
        legal_moves = self.get_legal_moves(board=board)
        # legal moves does not seem to be matchin what is in redis, find out why!!!
        #list should have move, mean,stdv
        move_stats = {}
        for move in legal_moves:
            move_scores = []
            cursor = '0'
            while cursor != 0:
                temp = f"'{move}'"
                match_value = "\\["+temp+"*"
                cursor, data = self.r.scan(cursor=cursor, match=match_value)

                for key in data:
                    # Fetch the value from the key
                    value = self.r.get(key)
                    move_scores.append(float(value))
                    # Check if the value matches your criteria
                    if value == 'your_desired_value':
                        print(f'Key: {key} - Value: {value}')
            mean = np.mean(move_scores)
            stdv = np.std(move_scores)
            move_stats[move] = [mean,stdv]
        return move_stats

    def pick_next_move(self,board: chess.Board):
        data = self.process_redis_boards()
        self.r.flushall()
        for i in range(0,len(data)-1):
            moves = data[i]
            score = float(data['predictions'][i])
            self.r.set(moves,score)
        move_stats = self.evaluate_redis(board)
        best_move = ''
        highest = ['move',0]

        for key in move_stats:
            score = move_stats[key][0] - move_stats[key][1]
            if score > highest[1]:
                highest[1] = score
                highest[0] = key
        return highest



