import chess
import pandas as pd

import chess
from Chess_Model.src.model.config.config import Settings
from  Chess_Model.src.model.classes.cnn_bb_scorer import boardCnnEval
from Chess_Model.src.model.classes.torch_model import model_operator


class move_picker():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()

        self.mdp = model_operator()
        self.mdp.load_model(model_path=s.torch_model_file)
        self.evaluator = boardCnnEval()
        # self.nnPredictionsCSV = s.nnPredictionsCSV


        # if "neuralNet" not in kwargs:
        #     raise Exception("No neural net supplied")
        # else:
        #     self.nn = kwargs["neuralNet"]
        


    def use_model(self,board:chess.Board):

        bitboards,metadata = self.evaluator.get_board_scores_applied(board=board)

        score = self.mdp.predict_single_example(bitboards=bitboards,metadata=metadata)

        return score

        
    

