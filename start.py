import chess
import csv
import redis
import hiredis
import ezboard




DEPTH = 3
move_dict = {}
r = redis.Redis(host='localhost', port=6379)


def get_legal_moves(board: ezboard.ezboard()):
    legal_moves=  [move.uci() for move in board.legal_moves]
    return(legal_moves)







def get_potential_boards(board: chess.Board()):
    boards = []
    temp_board = chess.Board(board.fen())
    result = get_legal_moves(board=board)
    for move in result:
        temp_board.push_uci(move)
        boards.append(temp_board)
        temp_board = chess.Board(board.fen())
    return(boards)

def get_all_boards(boards: list[chess.Board()]):
    total_boards = []
    tree = []
    for i in range(0,DEPTH):
        for board in boards:
            if i == (DEPTH-1):
                value = r.get(i)
                value
                for node in get_potential_boards(board=board):
                    total_boards.append(node.fen())
            else:    
                total_boards+=(get_potential_boards(board=board))
        boards = total_boards
        tree.append(total_boards)
    return total_boards

def store_all_moves_in_redis(board: chess.Board()):
    curr_moves = [0] * DEPTH
    start = get_legal_moves(board=board)
    for i in range(0,DEPTH):
        for move in start:
            curr_moves[i] = move
            if i == (DEPTH - 1):
                for moves in curr_moves:
                    tmp_board = board
                    tmp_board.push_uci(moves)



def get_moves(board: chess.Board, curr_moves=[], curr_depth=0):
    moves = get_legal_moves(board=board)
    for move in moves:
        curr_moves.append(move)
        temp_board = board
        temp_board.push_uci(move)
        if curr_depth != DEPTH - 1:
            curr_depth += 1
            get_moves(board=temp_board, curr_moves=curr_moves, curr_depth=curr_depth)
        else:
            redis_string = f"moves: {curr_moves} fen: {temp_board.fen()}"
            r.set(redis_string, 0)
            curr_moves.pop()
            curr_depth -= 1
            get_moves(board=temp_board, curr_moves=curr_moves, curr_depth=curr_depth)
        if curr_moves:
            curr_moves.pop()
            curr_depth -= 1
'''
#incomplete
def get_movesv2(board: chess.Board):
    curr_depth = 0
    curr_moves = []
    keys = {}
    while curr_depth != (DEPTH-1):
        moves = get_legal_moves(board=board)
        
        for move in moves:
            if curr_depth != (DEPTH-1):
                curr_moves.append(move)
                temp_board = board
                temp_board.push_uci(move)
                moves = get_legal_moves(board=temp_board)
            else:
                curr_moves.append(move)
                temp_board = board
                temp_board.push_uci(move)
                r.set(moves)     
        moves.pop()
'''
def get_movesv3(board: chess.Board):
    curr_moves = {}
    curr_depth = 0
    start_board = board
    moves = get_legal_moves(board=board) 
    while curr_depth < DEPTH:
        moves = get_legal_moves(board=board) 
        for move in moves:
            curr_moves[curr_depth] = move
            board = start_board
            for i in range(0,curr_depth):
                board.push_uci(curr_moves[i])
            board.push_uci(move)
            if curr_depth == (DEPTH-1):
                redis_string = ""
                moves_list = []
                temp = board
                for key in curr_moves:
                    moves_list.append(curr_moves[key])
                    temp.push_uci(curr_moves[key])
                redis_string = f"moves:{moves_list} fen:{temp.fen()}"
                r.set(redis_string,0)
                curr_depth -= 1
        
        curr_depth += 1

def get_movesv4(board: chess.Board):
    curr_moves = {}
    curr_depth = 0
    start_fen=board.fen()
    moves = get_legal_moves(board=board) 
    for move in moves:
        while curr_depth < DEPTH:
            curr_moves[curr_depth] = move
            temp_board = chess.Board(start_fen)
            for i in range(0,curr_depth+1):
                temp_board.push_uci(curr_moves[i])
            if curr_depth == (DEPTH-1):
                redis_string = ""
                moves_list = []
                temp_board = chess.Board(start_fen)
                for key in curr_moves:
                    moves_list.append(curr_moves[key])
                    temp_board.push_uci(curr_moves[key])
                redis_string = f"moves:{moves_list} fen:{temp_board.fen()}"
                print(redis_string)
                #r.set(redis_string,0)
                curr_depth -= 1

            curr_depth += 1           


def get_movesv5(board: ezboard.ezboard,curr_depth = 0):
    for move in get_legal_moves(board=board):
        board.move(move)
        if len(board.move_list) == (DEPTH):
            r.set(str(board.move_list),board.fen())
            board.go_back(num_of_moves=1)
        else:
            curr_depth += 1
            get_movesv5(board=board,curr_depth=curr_depth)


        
test_board = ezboard.ezboard()
get_movesv5(board=test_board)


