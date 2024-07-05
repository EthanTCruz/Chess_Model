import chess
from multiprocessing import Pool
import csv
import os
import pandas as pd
from sqlalchemy.orm import  Session
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from Chess_Model.src.model.classes.cnn_scorer import boardCnnEval
from Chess_Model.src.model.config.config import Settings
from Chess_Model.src.model.classes.sqlite.dependencies import  fetch_all_game_positions_rollup,get_rollup_row_count,board_to_GamePostition
from Chess_Model.src.model.classes.sqlite.models import GamePositions
from Chess_Model.src.model.classes.sqlite.database import SessionLocal

metadata_key = 'metadata'
bitboards_key = 'positions_data'
results_key = 'game_results'
feature_description = {
    'bitboards': tf.io.FixedLenFeature([], tf.string),
    'metadata': tf.io.FixedLenFeature([], tf.string),
    'target': tf.io.FixedLenFeature([], tf.string),
}

class game_analyzer:
    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        if "output_file" not in kwargs:
            self.output_file=s.scores_file
        else:
            self.output_file = kwargs["output_file"]
        

        if "persist_data" not in kwargs:
            self.persist_data = False
        else:
            self.persist_data = kwargs["persist_data"]

        self.evaluator = boardCnnEval()

        self.base_df = self.create_scores_df()
        self.recordsDataFile = s.recordsData
        
    def open_endgame_tables(self):
        self.evaluator.open_tables()

    def close_endgame_tables(self):
        self.evaluator.close_tables()
        
    def process_boards(self,total_moves:dict):
        
        data = self.create_scores_df()
        for key in total_moves.keys():

            values = key.split(":")
            moves = [values[0]]
            victor = total_moves[key][0]

            game = board_to_GamePostition(board= total_moves[key][1],victor=victor)
            scores = self.evaluate_board(game=game)          
            row = moves + list(scores.values())
            new_row = pd.DataFrame([row], columns=data.columns)
            data = pd.concat([data, new_row], ignore_index=True)
        
        return data

    @staticmethod
    def worker(game):
        if game is None: 
            return 1
        # This static method will be the worker function
        # Instantiate game_analyzer and evaluate_board here, since each process will have its own instance
        analyzer = game_analyzer()
        analyzer.refresh_evaluator(game=game)  # Ensure the evaluator is refreshed for the game
        return analyzer.evaluate_board(game=game)



    def create_csv(self):
        # Check if the file exists and remove it
        if os.path.exists(self.output_file) and not self.persist_data:
            os.remove(self.output_file)

        # Create a new CSV file with the column headers
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            features = self.evaluator.get_features()
            writer.writerow(features)
            pass

    def evaluate_board(self,game: GamePositions):
        self.refresh_evaluator(game=game)
        return self.evaluator.get_board_scores()
    
    def refresh_evaluator(self,game: GamePositions):
        self.evaluator.setup_parameters_gamepositions(game=game)



    def create_scores_df(self):
        scorer_obj = boardCnnEval()
        features = scorer_obj.get_features()
        features = ["moves(id)"] + features
        # Create a DataFrame with the specified column headers
        df = pd.DataFrame(columns=features)

        return df


    def process_sqlite_boards_pooling(self,db: Session = SessionLocal()):
        self.create_csv()

        row_count = get_rollup_row_count(db=db)
        games = fetch_all_game_positions_rollup(yield_size=500, db=db)

        # Initialize a process Pool
        with Pool(os.cpu_count()) as pool:
            # Map the worker function across the games, using tqdm for progress tracking
            results = list(tqdm(pool.imap_unordered(self.worker, games), total=row_count, desc="Processing Feature Data"))

        # Write the results to CSV
        with open(self.output_file, 'a', newline='') as gameEvalfile:
            writer = csv.writer(gameEvalfile)
            for scores in results:
                if scores:
                    if scores is None or scores == 1:
                        return 1
                    row = list(scores.values())
                    writer.writerow(row)

    def process_sqlite_boards(self,db: Session = SessionLocal()):
        self.create_csv()

        row_count = get_rollup_row_count(db=db)
        with open(self.output_file, 'a', newline='') as gameEvalfile:
            writer = csv.writer(gameEvalfile)
            batch = fetch_all_game_positions_rollup(yield_size=500, db=db)
            
            # Wrap the generator with tqdm
            for game in tqdm(batch, total=row_count, desc="Processing Feature Data"):
                try:
                    if game is None:
                        return 1

                    scores = self.evaluate_board(game=game)
                    row = list(scores.values())
                    writer.writerow(row)

                except Exception as e:
                    raise Exception(e)

    def process_single_board(self,board: chess.Board):
        df = self.base_df.copy()
        game = board_to_GamePostition(board=board)

        scores = self.evaluate_board(game=game)

        row = [board.fen()]+list(scores.values())

        new_row = pd.DataFrame([row], columns=df.columns)
        
        return new_row

    def create_records_csv(self):
            # Check if the file exists and remove it
        if os.path.exists(self.recordsDataFile):
            os.remove(self.recordsDataFile)

            # Create a new CSV file with the column headers
        with open(self.recordsDataFile, 'w', newline='') as recordsFile:
                pass
        
    def process_sqlite_boards_to_records(self, batch_size: int = 5):
        with tf.io.TFRecordWriter(Settings().recordsData) as writer:
            with SessionLocal() as db:
                row_count = get_rollup_row_count(db=db)
                batch = fetch_all_game_positions_rollup(yield_size=500, db=db)
                serialized_examples = []  # List to accumulate serialized examples
                for game in tqdm(batch, total=row_count, desc="Processing Feature Data"):
                    try:
                        if game:
                            self.evaluator.setup_parameters_gamepositions(game=game)
                            score = self.evaluator.get_board_scores_records()

                            serialized_data = serialize_data(score)
                            
                            features = {
                            'bitboards': _bytes_feature(serialized_data[0].numpy()),
                            'metadata': _bytes_feature(serialized_data[1].numpy()),
                            'target': _bytes_feature(serialized_data[2].numpy())
                            }
                            
                            serialized_data = tf.train.Example(features=tf.train.Features(feature=features))

                            serialized_examples.append(serialized_data)

                            # Check if we've accumulated enough examples to write a batch
                            if len(serialized_examples) >= batch_size:
                                for serialized_example in serialized_examples:
                                    writer.write(serialized_example.SerializeToString())
                                serialized_examples = []  # Reset the list after writing
                        else:
                            return 1
                    except Exception as e:
                        raise Exception(e)
                
                # Write any remaining examples after looping through all games
                for serialized_example in serialized_examples:
                    writer.write(serialized_example)
                    
    def set_feature_description(self):
        board = chess.Board()
        game = board_to_GamePostition(board=board)
        self.evaluator.setup_parameters_gamepositions(game=game)
        scores = self.evaluator.get_board_scores()
        self.feature_description = create_feature_description(data=scores)
        return self.feature_description
    
    def get_feature_description(self):
        return self.feature_description
    





def create_feature_description(data):
    feature_description = {}
    for key, value in data.items():
        if key in ['white mean', 'black mean', 'stalemate mean']:
            feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
        elif isinstance(value, int):
            feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
            # feature_description[key] = tf.io.FixedLenFeature([], tf.int64)
        elif isinstance(value, float):
            feature_description[key] = tf.io.FixedLenFeature([], tf.float32)
        elif isinstance(value, np.ndarray):
            # Arrays are serialized as strings, so we specify them as such
            feature_description[key] = tf.io.FixedLenFeature([], tf.string)
        else:
            raise ValueError(f"Unsupported data type: {type(value)} for key: {key}")
    return feature_description

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def serialize_tensor(tensor):
    return tf.io.serialize_tensor(tf.convert_to_tensor(tensor, dtype=tensor.dtype))


def split_board_scores(scores_dict: dict):
    # Collect all values within the nested metadata dictionary
    metadata = list(scores_dict[metadata_key].values())
    # metadata = []
    # for key, value in scores_dict['metadata'].items():
    #     metadata += [value]

    # For positions_data, it's a bit tricky since it looks like it might contain arrays.
    # Assuming you want to keep these arrays intact:
    bitboards = list(scores_dict[bitboards_key].values())
    
    # Collect all values within the game_results dictionary
    game_results = list(scores_dict[results_key].values())
    
    return bitboards, metadata, game_results

def serialize_data(scores_dict):

        bb,md,gr = split_board_scores(scores_dict)
        
        bitboards_tensor = tf.stack([tf.convert_to_tensor(board, dtype=tf.int8) for board in bb])
        serialized_bitboards = tf.io.serialize_tensor(bitboards_tensor)
        
        metadata_tensor = tf.convert_to_tensor(md,dtype=tf.float16)
        serialized_metadata = serialize_tensor(metadata_tensor)
        
        target_tensor = tf.convert_to_tensor(gr,dtype=tf.float16)
        serialized_target = serialize_tensor(target_tensor)
        
        return serialized_bitboards, serialized_metadata, serialized_target