import chess
import chess.pgn
import chess.engine
import random
import time
from math import log,sqrt,e,inf
import ast
from Chess_Model.src.model.classes.nn_model import neural_net 



class node():
    def __init__(self):
        self.state = chess.Board()
        self.action = ''
        self.children = set()
        self.parent = None
        self.N = 0
        self.n = 0
        self.v = 0

class mcts():

    def __init__(self,neuralNet: neural_net) -> None:
        self.nn = neuralNet

    def ucb1(self,curr_node):
        ans = curr_node.v+sqrt(2)*(sqrt(log(curr_node.N+e+(10**-6))/(curr_node.n+(10**-10))))
        return ans

    def rollout(self,curr_node,max_depth: int = 10,cnn:bool = False):
        depth = 0
        if(curr_node.state.is_game_over()):
            board = curr_node.state
            if(board.result()=='1-0'):
                #print("h1")
                return (1.5,curr_node)
            elif(board.result()=='0-1'):
                #print("h2")
                return (-1.5,curr_node)
            else:
                return (0,curr_node)
        
        while not curr_node.state.is_game_over() and depth < max_depth:
            all_moves = list(curr_node.state.legal_moves)
            if not all_moves:
                break  # No legal moves, break the loop

            selected_move = random.choice(all_moves)
            tmp_state = curr_node.state.copy()
            tmp_state.push(selected_move)

            child = node()  # Create a new node
            child.state = tmp_state
            child.parent = curr_node
            curr_node.children.add(child)  # In rollout, parent nodes are not tracked

            depth += 1  # Increment the depth

        # Evaluate the board state at the final depth
        # The evaluation function needs to be defined based on your requirements
        if cnn:
            score = self.nn.score_board_cnn(curr_node.state)
            score = score['white'] - score['black']
            score = 4 * score.values[0]
            
        else:
            score = self.nn.score_board(curr_node.state)
            if 0.5 - 0.1 <= score <= 0.5 + 0.1:
                #trying best to get rid of stalemates
                score = 0
            else:
                #mapping [0,1] to [-1,1]
                score = (-1 + (2 * round(score,0)))

        return score, curr_node

    def expand(self,curr_node,white):
        if(len(curr_node.children)==0):
            return curr_node
        max_ucb = -inf
        if(white):

            max_ucb = -inf
            sel_child = None
            for i in curr_node.children:
                tmp = self.ucb1(i)
                if(tmp>max_ucb):

                    max_ucb = tmp
                    sel_child = i

            return(self.expand(sel_child,0))

        else:

            min_ucb = inf
            sel_child = None
            for i in curr_node.children:
                tmp = self.ucb1(i)
                if(tmp<min_ucb):

                    min_ucb = tmp
                    sel_child = i

            return self.expand(sel_child,1)

    def rollback(self,curr_node,reward):
        curr_node.n+=1
        curr_node.v+=reward
        while(curr_node.parent!=None):
            curr_node.N+=1
            curr_node = curr_node.parent
        return curr_node

    def mcts_pred(self,curr_node,over,white,preferred_moves: list = None,iterations: int = 10,max_depth: int = 10,cnn:bool = False):
        if(over):
            return 0
        if preferred_moves:
            all_moves = []
            for i in preferred_moves:
                for move in ast.literal_eval(i):
                    all_moves.append(curr_node.state.uci(curr_node.state.parse_uci(move)))
        else:
            all_moves = [curr_node.state.uci(i) for i in list(curr_node.state.legal_moves)]
        map_state_move = dict()
        
        for i in all_moves:
            tmp_state = chess.Board(curr_node.state.fen())
            tmp_state.push_uci(i)
            child = node()
            child.state = tmp_state
            child.parent = curr_node
            curr_node.children.add(child)
            map_state_move[child] = i
            
        while(iterations>0):
            if(white):

                max_ucb = -inf
                sel_child = None
                for i in curr_node.children:
                    tmp = self.ucb1(i)
                    if(tmp>max_ucb):

                        max_ucb = tmp
                        sel_child = i
                ex_child = self.expand(sel_child,0)
                reward,state = self.rollout(curr_node=ex_child,max_depth=max_depth,cnn=cnn)
                curr_node = self.rollback(state,reward)
                iterations-=1
            else:

                min_ucb = inf
                sel_child = None
                for i in curr_node.children:
                    tmp = self.ucb1(i)
                    if(tmp<min_ucb):

                        min_ucb = tmp
                        sel_child = i

                ex_child = self.expand(sel_child,1)

                reward,state = self.rollout(ex_child,max_depth=max_depth,cnn=cnn)

                curr_node = self.rollback(state,reward)
                iterations-=1
        if(white):
            
            mx = -inf

            selected_move = ''
            for i in (curr_node.children):
                tmp = self.ucb1(i)
                if(tmp>mx):
                    mx = tmp
                    selected_move = map_state_move[i]
            return selected_move
        else:
            mn = inf

            selected_move = ''
            for i in (curr_node.children):
                tmp = self.ucb1(i)
                if(tmp<mn):
                    mn = tmp
                    selected_move = map_state_move[i]
            return selected_move



    def clear(self,node: node):
        if node.children is None:
            new_node = node.parent
            node = None
            self.clear(node=new_node)
        else:
            for child in node.children:
                self.clear(node=child)
            if node.parent is None:
                node = None
                del node
            return 0
        





    def mcts_best_move(self,board: chess.Board,preferred_moves: list = None,cnn:bool = False,iterations=100,max_depth: int = 10):
        if len(preferred_moves) == 1:
            if not cnn:
                return ast.literal_eval(preferred_moves[0])[0]
            else:
                return ast.literal_eval(preferred_moves.values[0])[0]
        root = node()
        root.state = board
        is_white_to_move = board.turn


        best_move_uci = self.mcts_pred(curr_node=root,over=board.is_game_over(),
                                white=is_white_to_move,preferred_moves=preferred_moves,
                                    iterations=iterations,max_depth=max_depth,cnn=cnn)
        if best_move_uci[-1] in ['b','r','n']:
            best_move_uci = best_move_uci[:-1] + 'q'

        self.clear(node=root)

        return(best_move_uci)

