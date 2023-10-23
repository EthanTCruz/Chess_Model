import chess
import redis
import hiredis
from config import Settings






class populator():

    def __init__(self,**kwargs):
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        if "score_depth" not in kwargs:
            self.score_depth=2
        else:
            self.score_depth = kwargs["score_depth"]
        if "mate_depth" not in kwargs:
            self.mate_depth=3
        else:
            self.mate_depth = kwargs["mate_depth"]        

        if "board" not in kwargs:
            raise Exception("Need initial baord")
        else:
            self.board = kwargs["board"]
        
        if "pgn" not in kwargs:
            self.predictions_board='predictions.csv'
        else:
            self.pgn = kwargs["pgn"]
            if kwargs["victor"] is not None:
                raise Exception("Need victor color")
            else:
                self.victor = kwargs["victor"]

        if "redis_score_db" not in kwargs:
            self.r_score = redis.Redis(host='localhost', port=6379,db=1)
        else:
            self.r_score = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_score_db))
        
        if "redis_mate_db" not in kwargs:
            self.r_mate = redis.Redis(host='localhost', port=6379,db=2)
        else:
            self.r_mate = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_mate_db))



    def reset_and_fill_redis(self):
        self.r_score.flushall()
        self.populate_redis(board=self.board,moves=self.get_legal_moves(board=self.board))



    def get_legal_moves(self,board: chess.Board):
        legal_moves=  [move.uci() for move in board.legal_moves]
        return(legal_moves)
        

    def populate_redis(self,board: chess.Board, moves: list[str] = [],initial_movestack_length: int = 0):
        move_dict = {}
        if initial_movestack_length == 0:
            initial_movestack_length = len(board.move_stack)

        for move in moves:
            try:
                board.push_uci(move)
                value = "u"

                if board.is_checkmate():
                    #black win
                    value = "w"
                    if board.turn:
                        #white win
                        value = "b"
                    move_list = [move.uci() for move in board.move_stack[initial_movestack_length:]]
                    self.r_score.set(f'{str(move_list)}:{ board.fen()}',value)
                    #basing off model will only be run on my turn, and 0 signifies losing checkmate
                    value = 1
                    self.r_mate.set(f'{str(move_list)}',value)
   
                score_move_list = [move.uci() for move in board.move_stack[(-self.score_depth):]]
                mate_move_list = [move.uci() for move in board.move_stack[(-self.mate_depth):]]
                if board.is_stalemate():
                    #stalemate
                    value = "s"
                    self.r_score.set(f'{str(score_move_list)}:{ board.fen()}',value)

                    #auto sets values for stalemate to 0.5
                    self.r_mate.set(f'{str(mate_move_list)}',0.5)
                current_depth = (len(board.move_stack) - initial_movestack_length)
                if current_depth > self.score_depth -1 and current_depth < self.mate_depth:
                        self.r_score.set(f'{str(score_move_list)}:{ board.fen()}',value)
                if current_depth > self.mate_depth -1 :
                    self.r_mate.set(f'{str(mate_move_list)}',value)               
                else:
                    legal_moves = self.get_legal_moves(board)
                    if legal_moves:
                        sub_dict = self.populate_redis(board,moves=legal_moves,initial_movestack_length=initial_movestack_length)
                        move_dict.update(sub_dict)
            except ValueError:
                pass
            board.pop()
        return move_dict
    
        

