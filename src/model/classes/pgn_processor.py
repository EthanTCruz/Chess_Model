import chess.pgn
import csv
from Chess_Model.src.model.classes.sqlite.dependencies import insert_bulk_boards_into_db
from Chess_Model.src.model.classes.sqlite.models import GamePositions
from tqdm import tqdm
from Chess_Model.src.model.classes.sqlite.database import SessionLocal
from sqlalchemy.orm import  Session
import os

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

    def pgn_fen_to_sqlite(self,db: Session = SessionLocal()):
        total_games = 0
        for filename in os.listdir(self.pgn_file):
            file = f"{self.pgn_file}{filename}"
            total_games += count_games_in_pgn(pgn_file=file)
        for filename in os.listdir(self.pgn_file):
            file = f"{self.pgn_file}{filename}"
            with open(file) as pgn:
                for _ in tqdm(range(total_games), desc="Processing Games to DB"):
                    game = chess.pgn.read_game(pgn)

                    if game is None:
                        break  # end of file
                    board = game.board()
                    board_victors = []
                    victor = 'NA'
                    if game.headers["Result"] == '1-0':
                        victor = 'w'
                    elif game.headers["Result"] == '0-1':
                        victor = 'b'
                    elif game.headers["Result"] == '1/2-1/2':
                        victor = 's'

                    for move in game.mainline_moves():
                        board.push(move=move)
                        board_victors.append((board.copy(), victor))
                    insert_bulk_boards_into_db(board_victors=board_victors, db=db)


def count_games_in_pgn(pgn_file):
    count = 0
    with open(pgn_file) as pgn:
        while chess.pgn.read_game(pgn) is not None:
            count += 1
    return count


