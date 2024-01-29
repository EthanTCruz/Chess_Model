import chess.pgn
import csv
from Chess_Model.src.model.classes.sqlite.dependencies import *
from Chess_Model.src.model.classes.sqlite.models import GamePositions
from tqdm import tqdm
from Chess_Model.src.model.classes.sqlite.database import SessionLocal
class pgn_processor():
    def __init__(self,pgn_file,csv_file) -> None:
        self.pgn_file = pgn_file
        self.csv_file = csv_file
    

    def pgn_fen_to_csv(self,victor="NA"):
        pgn = open(self.pgn_file)
        
        with open(self.csv_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    while True:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            break  # end of file
                        board = game.board()
                        victor = 's'
                        if game.headers["Result"] == '1-0':
                            victor = 'w'
                        elif game.headers["Result"] == '0-1':
                            victor = 'b'
                        if victor != 's':
                            move_list = []
                            for move in game.mainline_moves():
                                move_list.append(move)
                                board.push(move=move)
                                row = ''
                                #only want to proccess winning player moves
                                if victor != 'NA':
                                    if victor == 'w' and not board.turn:
                                        row = [str([move.uci() for move in board.move_stack]),board.fen(),victor]
                                        writer.writerow(row)
                                    elif victor =='b' and  board.turn:
                                        row = [str([move.uci() for move in board.move_stack]),board.fen(),victor]
                                        writer.writerow(row)
                                else:
                                    row = [str([move.uci() for move in board.move_stack]),board.fen(),victor]
                                    writer.writerow(row)   

    def pgn_fen_to_sqlite(self):
        with SessionLocal() as db:
            total_games = count_games_in_pgn(pgn_file=self.pgn_file)
            with open(self.pgn_file) as pgn:
                for _ in tqdm(range(total_games), desc="Processing games"):
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break  # end of file
                    board = game.board()
                    victor = 'NA'
                    if game.headers["Result"] == '1-0':
                        victor = 'w'
                    elif game.headers["Result"] == '0-1':
                        victor = 'b'
                    elif game.headers["Result"] == '1/2-1/2':
                        victor = 's'

                    for move in game.mainline_moves():
                        board.push(move=move)
                        insert_board_into_db(board=board,victor=victor,db=db)


def count_games_in_pgn(pgn_file):
    count = 0
    with open(pgn_file) as pgn:
        while chess.pgn.read_game(pgn) is not None:
            count += 1
    return count


