from Chess_Model.src.model.classes.board_analyzer import board_analyzer
from Chess_Model.src.model.classes.MCTS import mcts
import chess
import random


class move_picker():
    def __init__(self,ucb_constant:float = None,scores: list = [1.5,-1.5,0]) -> None:
        ba = board_analyzer()
        self.mcts = mcts(board_analyzer=ba,
                         ucb_constant=ucb_constant,
                         scores=scores)
        pass
        
    def get_best_move(self,ucb_constant: float,
                      scores: list,
                      board:chess.Board,
                      iterations:int = 100):
        random.seed(3141)
        self.mcts.set_ucb(ucb = ucb_constant)
        self.mcts.set_scores(scores=scores)
        return self.mcts.mcts_best_move(board=board,iterations=iterations)


