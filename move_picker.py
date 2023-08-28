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
from config import Settings

class move_picker():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        if "redis_score_db" not in kwargs:
            self.r_score = redis.Redis(host='localhost', port=6379,db=1)
        else:
            self.r_score = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_score_db))

    def get_legal_moves(self,board: chess.Board):
        legal_moves =  [move.uci() for move in board.legal_moves]
        return(legal_moves)

    def average_and_stdv_scores(self,board: chess.Board):
        legal_moves = self.get_legal_moves(board=board)
        move_stats = {}
        for move in legal_moves:
            move_scores = []
            cursor = '0'
            while cursor != 0:
                temp = f"'{move}'"
                match_value = "\\["+temp+"*"
                cursor, data = self.r_score.scan(cursor=cursor, match=match_value)

                for key in data:
                    # Fetch the value from the key
                    value = self.r_score.get(key)
                    move_scores.append(float(value))
                    # Check if the value matches your criteria
                    if value == 'your_desired_value':
                        print(f'Key: {key} - Value: {value}')
            mean = np.mean(move_scores)
            stdv = np.std(move_scores)
            move_stats[move] = [mean,stdv]
        return move_stats

    def highest_average_move(self,board: chess.Board):
        move_stats = self.average_and_stdv_scores(board)
        best_move = ''
        highest = ['move',0]
        for key in move_stats:
            score = move_stats[key][0] - move_stats[key][1]
            if score > highest[1]:
                highest[1] = score
                highest[0] = key
        return highest

