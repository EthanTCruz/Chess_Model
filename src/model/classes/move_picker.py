import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from src.model.classes.game_analyzer import game_analyzer
import numpy as np
from src.model.classes.redis_populator import populator
import redis
import os
import chess
from src.model.config.config import Settings
import ast

class Node:
    def __init__(self, move=None,mate_depth = 3):
        self.move = move
        self.children = {}
        self.is_checkmate = None
        self.mate_depth = mate_depth


class move_picker():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        if "redis_score_db" not in kwargs:
            self.r_score = redis.Redis(host='localhost', port=6379,db=1)
        else:
            self.r_score = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_score_db))
        if "player" not in kwargs:
            self.player = 'w'
        else:
            self.player = kwargs['player']
        if "redis_mate_db" not in kwargs:
            self.r_mate = redis.Redis(host='localhost', port=6379,db=2)
        else:
            self.r_mate = redis.StrictRedis(host=s.redis_host, port=s.redis_port,db=int(s.redis_mate_db), charset="utf-8", decode_responses=True)
 

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

    def find_forced_wins(self,board: chess.Board):
        board = chess.Board(fen=board.fen())
        cursor = 0
        count = 1000
        initial_moves = self.get_legal_moves(board=board)
        for move in initial_moves:
            while True:
                temp = f"'{move}'"
                match_value = "\\["+temp+"*"

                cursor, keys = self.r_score.scan(cursor=cursor, count=count,match=match_value)
                # Print the retrieved keys
                for key in keys:
                    # go through move and delte all keys with start move if no win routes are found
                    if self.is_win_route(key=move):
                        return key


                    # Break the loop if the cursor returns to the start
                    if cursor == 0:
                        break
                    cursor = 0        
        return 0

    def is_win(self,key):
        victor = self.r_mate.get(key)
        if victor == self.player:
            return True
        return False

    def is_win_route(self,key):
        if self.is_win(key=key):
            return self.check_prev_branch(key=key)
        else:
            return False

    def check_prev_branch(self,key):
        #slice off last two moves to get to  next pro moves
        moves = list(key)[0:-2]
        if moves == 1:
            return key
        cursor = 0
        count = 1000

        while True:
            #removes ending bracket
            temp = str(moves)[1:-1]
            match_value = "\\["+temp+",*"

            cursor, keys = self.r_score.scan(cursor=cursor, count=count,match=match_value)
            # Print the retrieved keys
            for key in keys:
                # go through move and delte all keys with start move if no win routes are found
                if not self.is_win_route(key=key):
                    
                    return key


                # Break the loop if the cursor returns to the start
                if cursor == 0:
                    break            

    def find_wins(self,board: chess.Board):
        board = chess.Board(fen=board.fen())
        initial_moves = self.get_legal_moves(board=board)
        for initial_move in initial_moves:
            moves_to_check = self.get_legal_moves(board=board)

            board.push_uci(initial_move)
            for move in moves_to_check:
                temp = initial_move + move
                temp = str(initial_move)[1:-1]
                match_value = "\\["+temp+",*"
                keys_to_check = self.r_mate.keys(match_value=match_value)
                   
    def insert_moves(self,root, moves, is_checkmate):
        current = root
        for move in moves:
            if move not in current.children:
                current.children[move] = Node(move)
            current = current.children[move]
        current.is_checkmate = is_checkmate

    def create_move_tree(self):
        root = Node()
        for key in self.r_mate.keys():
            moves = list(key)  # assuming moves are comma-separated in redis
            value = False
            if self.r_mate.get(key) == self.player:
                value = True
            is_checkmate = value
            self.insert_moves(root, moves, is_checkmate)
        return root

    def pro_even_compress(self):
        cursor = 0
        count = 1000
        while True:
            # Scan the Redis database using the cursor
            cursor, keys = self.r_mate.scan(cursor=cursor, count=count)
            # Print the retrieved keys
            for key in keys:

                moves = ast.literal_eval(key)
                if (len(moves) % 2 == 0) and (len(moves) > 1):
                    if self.r_mate.get(key) == '1':
                            self.compress_branch_to_value(branch=key,score=1)
                            break
                    if cursor == 0:
                            self.compress_branch_to_value(branch=key,score=0)
                            break
            # Break the loop if the cursor returns to the start
            if cursor == 0:
                    break

    def opp_odd_compress(self):
        cursor = 0
        count = 1000
        while True:
            # Scan the Redis database using the cursor
            cursor, keys = self.r_mate.scan(cursor=cursor, count=count)
            # Print the retrieved keys
            for key in keys:
                moves = ast.literal_eval(key)
                if (len(moves) % 2 == 1)  and (len(moves) > 1):
                    if self.r_mate.get(key) == '0':

                            self.compress_branch_to_value(branch=key,score=0)
                            break
                    if cursor == 0:
                            self.compress_branch_to_value(branch=key,score=0)
                            break
            # Break the loop if the cursor returns to the start
            if cursor == 0:
                    break

    def compress_branch_to_value(self,branch,score):
        cursor = 0
        count = 1000
        moves = ast.literal_eval(branch)
        temp = str(moves[:-1])[1:-1]
        match_value = f"*\\{temp[1:-1]}*"
        while True:

            # Scan the Redis database using the cursor
            cursor, keys = self.r_mate.scan(cursor=cursor, count=count,match=match_value)

            for key in keys:
                if len(ast.literal_eval(key)) != 1:
                    self.r_mate.delete(key)

            # Break the loop if the cursor returns to the start
            if cursor == 0:
                    self.r_mate.set(str(moves[:-1]),score)
                    break


    def find_forced_wins(self,board: chess.Board):
        board = chess.Board(fen=board.fen())
        initial_moves = self.get_legal_moves(board=board)
        while True:
            if (len(initial_moves) == (self.r_mate.dbsize())):
                for key in self.r_mate.keys():
                    if self.r_mate.get(key) == '1':
                        return key
                return 0
            else:
                self.compress(evenMode=False)
                self.compress(evenMode=True)


    def compress(self,evenMode = True):
        eval = {}
        #value for evaluating whether even or odd
        eval['parity'] = 1
        #value searching for in branch
        eval['success'] = ["1"]
        #default value for if no success found
        eval['fail'] = 0

        if evenMode:
            #value for evaluating whether even or odd
            eval['parity'] = 0
            #value searching for in branch
            eval['success'] = [0,0.5,'u']
            #default value for if no success found
            eval['fail'] = 1

        cursor = 0
        count = 1000
        continueEval = True
        while continueEval:

            # Scan the Redis database using the cursor
            cursor, keys = self.r_mate.scan(cursor=cursor, count=count)
            for key in keys:
                moves = ast.literal_eval(key)
                if len(moves) == 1:
                    pass
                #condition for if key is of correct parity
                elif (len(moves) % 2 == eval['parity']):
                    if str(self.r_mate.get(key)) in  map(str, eval['success']):
                            self.compress_branch_to_value(branch=key,score=eval['success'][0])
                else:
                    if self.r_mate.exists(key):
                        self.compress_branch_to_value(branch=key,score=eval['fail'])
            if cursor == 0:
                continueEval = False
                break




