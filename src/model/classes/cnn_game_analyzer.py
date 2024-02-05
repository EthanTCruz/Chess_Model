import chess

import csv
import os
import pandas as pd
from sqlalchemy.orm import  Session
from Chess_Model.src.model.classes.cnn_scorer import boardCnnEval
from tqdm import tqdm
from Chess_Model.src.model.config.config import Settings
from Chess_Model.src.model.classes.sqlite.dependencies import  fetch_all_game_positions_rollup,get_rollup_row_count,board_to_GamePostition
from Chess_Model.src.model.classes.sqlite.models import GamePositions
from Chess_Model.src.model.classes.sqlite.database import SessionLocal

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
