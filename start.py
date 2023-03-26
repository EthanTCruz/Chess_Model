import chess
import csv
import redis
import hiredis
import ezboard
import sys




DEPTH = 2
move_dict = {}
r = redis.Redis(host='localhost', port=6379)


def get_legal_moves(board: ezboard.ezboard):
    legal_moves=  [move.uci() for move in board.legal_moves]
    return(legal_moves)
       

'''
def get_movesv5(board: ezboard.ezboard,moves = []):
    start = board.fen()
    for i in range(0, len(moves)):
        board.move(moves[i])
        if len(board.move_list) > (DEPTH):
            print(f"DEPTH reached on {board.move_list}")
            r.set(str(board.move_list),board.fen())
            board.go_back(num_of_moves=1)
            if len(moves) > 0:
                get_movesv5(board=ezboard.ezboard(fen=board.fen(),move_list=board.move_list), moves=moves[i+1:len(moves)])

        else:
            print("else")
            get_movesv5(board=ezboard.ezboard(fen=board.fen(),move_list=board.move_list), moves=get_legal_moves(board=board))
        board.go_back(num_of_moves=1)
'''

def get_movev5(board: ezboard.ezboard, moves: list[str] = [], depth: int = DEPTH):
    move_dict = {}
    for move in moves:
        try:
            board.push_uci(move)
            if len(board.move_stack) > depth:
                print(f"DEPTH reached on {str([move.uci() for move in board.move_stack])}")
                r.set(str([move.uci() for move in board.move_stack]), board.fen())
            else:
                legal_moves = get_legal_moves(board)
                if legal_moves:
                    sub_dict = get_movev5(board, legal_moves, depth)
                    move_dict.update(sub_dict)
        except ValueError:
            pass
        board.pop()
    return move_dict
        
test_board = ezboard.ezboard()
get_movev5(board=test_board,moves=get_legal_moves(board=test_board))

#"rnbqkbnr/ppppp1pp/8/5p2/8/2P3P1/PP1PPP1P/RNBQKBNR b KQkq - 0 2"
#"rnbqkbnr/ppppp1pp/8/5p2/8/2P3P1/PP1PPP1P/RNBQKBNR b KQkq - 0 2"
#rint(get_all_boards(boards=[test_board]))