import chess
import pandas as pd

import chess
from Chess_Model.src.model.config.config import Settings
from  Chess_Model.src.model.classes.cnn_bb_scorer import boardCnnEval



class move_picker():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()

        self.evaluator = boardCnnEval()
        # self.nnPredictionsCSV = s.nnPredictionsCSV


        # if "neuralNet" not in kwargs:
        #     raise Exception("No neural net supplied")
        # else:
        #     self.nn = kwargs["neuralNet"]
        


    def use_model(self,board:chess.Board):
        self.evaluator.setup_parameters_board(board=board)
        scores = self.evaluator.get_board_scores_applied()
        return scores

        
    

