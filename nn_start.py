import numpy as np
import tensorflow as tf
from tensorflow import keras
import chess
import redis
import hiredis

'''
r = redis.Redis(host='localhost', port=6379)

# Set the initial cursor position
cursor = 0

# Set the number of keys to retrieve at a time
count = 10

while True:
    # Scan the Redis database using the cursor
    cursor, keys = r.scan(cursor=cursor, count=count)
    # Print the retrieved keys
    for key in keys:
        print(key.decode('utf-8'))

    # Break the loop if the cursor returns to the start
    if cursor == 0:
        break
'''

Start_value =  "['d2d4', 'e7e6', 'c1h6']:rnbqkbnr/pppp1ppp/4p2B/8/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2"
fen = Start_value.split(":")[1]
board = chess.Board(fen=fen)
fen_components = fen.split(" ")
# 0 = positions , 1 = turn , 2 = castling abilities , 3 = en passant targets , 4 = 50 move counter , 5 = number of moves
attackers = board.attackers(chess.WHITE,chess.H3)
print("done")

def get_piece_amounts(board, team):
    #board = fen.split(" ")[0]
    pieces = ['P','N','B','R','Q']
    if team == 'b':
        pieces = ['p','n','b','r','q']
    for piece in pieces:
        results.append(board.count(piece))
    results = []
    return results

def get_game_time(board):
    #could be split up into two features for each side?
    count = len(board)
    count = count - board.count("/")
    for i in range(1,9):
        count = count - board.count(str(i))
    return (round(count/32))

def is_checkmate(board: chess.Board):
    if board.is_checkmate():
        return 1
