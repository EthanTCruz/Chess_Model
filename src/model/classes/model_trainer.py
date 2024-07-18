from Chess_Model.src.model.classes.pgn_processor import pgn_processor

from Chess_Model.src.model.classes.pretorch_files.game_analyzer import game_analyzer

from Chess_Model.src.model.classes.pretorch_files.potential_board_populator import populator

import os
import chess
from Chess_Model.src.model.config.config import Settings
import random
from Chess_Model.src.model.classes.cnn_move_picker import move_picker

import chess.pgn
import ast
import cowsay
import csv







class trainer():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        self.self_play_pgn = s.self_play
        self.selfPlayModel = s.SelfPlayModelFilename
        self.scores_file = s.scores_file
        self.games_csv_file = s.games_csv_file
        self.simGames = s.selfTrainBaseMoves


        if "neuralNet" not in kwargs:
            raise Exception("No neural net supplied")
        else:
            self.nn = kwargs["neuralNet"]

        if "redis_score_db"  in kwargs:
            self.r_score = kwargs["redis_score_db"]
            self.mp = move_picker(redis_score_db=self.r_score,
                neuralNet=self.nn)
        else:
            self.mp = move_picker(neuralNet=self.nn)





    def self_train(self,iterations: int = 1,depth: int = 1,trainModel: bool = True):
        for i in range(0,iterations):
            self.create_blank_pgn()
            board = chess.Board()

            pop = populator(score_depth = depth,board=board)
            pop.get_all_moves()

            move_list = pop.return_total_moves().keys()
            
            del pop

            for key in move_list:
                    values = key.split(":")
                    moves = values[0]
                    moves = ast.literal_eval(moves)
                    self.gameplay(moves=moves)
                    iterations += 1
            if trainModel:
                self.train_on_self_play()


            
    def train_on_self_play(self):
        if os.path.exists(self.scores_file):
            os.remove(self.scores_file)
        if os.path.exists(self.games_csv_file):
            os.remove(self.games_csv_file)




        gam_an_obj = game_analyzer(scores_file=self.scores_file)
        pgn_obj = pgn_processor(pgn_file=self.self_play_pgn,csv_file=self.games_csv_file)

        cowsay.cow(f"Converting pgn file to csv: {self.games_csv_file}")    
        pgn_obj.pgn_fen_to_csv()
        cowsay.cow(f"Generating feature data from pgn boards in csv: {self.scores_file}")
        gam_an_obj.process_csv_boards(csv_file=self.games_csv_file)

        
        cowsay.cow("Create neural net")
        
        cowsay.cow(f"Training neural net on {self.scores_file} and saving weights to {self.selfPlayModel}")

                
        self.nn.create_and_evaluate_model_batch()



    def gameplay(self,moves):
        game = chess.pgn.Game()
        node = game.add_variation(chess.Move.from_uci(moves[0]))
        board = chess.Board()
        board.push(chess.Move.from_uci(moves[0]))
        if len(moves) > 1:
            for move in moves[1::]:
                node = node.add_variation(chess.Move.from_uci(move))
                board.push(chess.Move.from_uci(move))

        while True:
            if(board.is_game_over()):
                break
            else:
                move = self.mp.use_model_timed(board=board,time_limit=10)

                if move in board.legal_moves:
                    board.push(move)
                    node = node.add_variation(move)
                else:
                    raise ValueError(f"Illegal move {move} generated by model")

        game.headers["Result"] = board.result()
        #50 move draw error, need to score draws as losses
        with open(self.self_play_pgn, "a", encoding="utf-8") as pgn_file:
            exporter = chess.pgn.FileExporter(pgn_file)
            game.accept(exporter)
        del game
        del board

        return 0
    


    def create_blank_pgn(self):
        with open(self.self_play_pgn, 'a') as file:
            pass 
