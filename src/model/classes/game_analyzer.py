import chess

import csv
import os
import pandas as pd
from Chess_Model.src.model.classes.scorer import boardEval
from tqdm import tqdm
from Chess_Model.src.model.config.config import Settings

Start_value =  "['d2d4', 'e7e6', 'c1h6']:rnbqkbnr/pppp1ppp/4p2B/8/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2"

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
        self.evaluator = boardEval()
        self.evaluator.open_tables()

        self.basedf = self.create_scores_df()






    def process_single_board(self,board_key,victor="NA"):
        
        with open(self.output_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    values = board_key.split(":")
                    moves = [values[0]]
                    fen = values[1]
                    scores = list(self.evaluate_board(fen=fen,victor=victor).values())
                    row = moves + scores
                    writer.writerow(row)
        

    def process_single_fen(self,fen,victor="NA"):
        
        self.create_csv()
        with open(self.output_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    scores = list(self.evaluate_board(fen=fen,victor=victor).values())
                    row = [fen] + scores
                    writer.writerow(row)
        

    def process_single_fen_non_csv(self,fen,victor="NA"):
        df = self.basedf.copy()
        



        scores = list(self.evaluate_board(fen=fen,victor=victor).values())
        row = [fen] + scores

        new_row = pd.DataFrame([row], columns=df.columns)
        
        return new_row

    
    def process_csv_boards(self,csv_file):
        
        self.create_csv()
        with open(csv_file) as f:
            total_lines = sum(1 for line in f)
        with open(self.output_file, 'a', newline='') as gameEvalfile:
            writer = csv.writer(gameEvalfile)
            # find total number of lines in the file
            with open(csv_file, 'r') as fenfile:
                csv_reader = csv.reader(fenfile)
                for row in tqdm(csv_reader, total=total_lines):
                        if row == "":
                             return 1
                        try:
                            victor = row[2]
                            fen = row[1]
                            moves = [row[0]]
                            scores = list(self.evaluate_board(fen=fen,victor=victor).values())

                            #row = [moves,scores]
                            moves += scores
                            writer.writerow(moves)
                            pass
                        except AttributeError:
                            print(row) 
                            raise Exception("here")
        

    def process_boards(self,total_moves:dict):
        
        self.create_csv()
        with open(self.output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for key in total_moves.keys():
                values = key.split(":")
                moves = [values[0]]
                fen = values[1]
                scores = self.evaluate_board(fen=fen,board = total_moves[key][1])
                value = total_moves[key][0]
                if value == 'b':
                        scores["w/b"] = 'b'
                elif value == 'w':
                    scores["w/b"] = 'w'
                elif value == 's':
                    scores["w/b"] = 's'          
                row = moves + list(scores.values())
                writer.writerow(row)
        

    def process_boards_non_csv(self,total_moves:dict):
        
        data = self.create_scores_df()
        for key in total_moves.keys():
            values = key.split(":")
            moves = [values[0]]
            fen = values[1]
            scores = self.evaluate_board(fen=fen,board = total_moves[key][1])
            value = total_moves[key][0]
            if value == 'b':
                    scores["w/b"] = 'b'
            elif value == 'w':
                scores["w/b"] = 'w'
            elif value == 's':
                scores["w/b"] = 's'          
            row = moves + list(scores.values())
            new_row = pd.DataFrame([row], columns=data.columns)
            data = pd.concat([data, new_row], ignore_index=True)
        
        return data


    def create_scores_df(self):
        scorer_obj = boardEval()
        features = scorer_obj.get_features()

        # Create a DataFrame with the specified column headers
        df = pd.DataFrame(columns=features)

        return df




    def csv_has_data_besides_header(self):
        with open(self.output_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the header
            next(reader, None)
            # Check if there's another row
            return any(reader) 
    
    def evaluate_board(self,fen,victor="NA",board: chess.Board = None):
        self.refresh_evaluator(fen=fen,board=board)
        return self.evaluator.get_board_scores(victor=victor)
    
    def refresh_evaluator(self,fen,board: chess.Board = None):
        self.evaluator.setup_parameters(board=board,fen=fen)

    def create_csv(self):
        # Check if the file exists and remove it
        if os.path.exists(self.output_file) and not self.persist_data:
            os.remove(self.output_file)

        # Create a new CSV file with the column headers
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            features = self.evaluator.get_features()
            writer.writerow(features)

