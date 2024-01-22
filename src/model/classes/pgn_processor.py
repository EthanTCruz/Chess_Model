import chess.pgn
import csv


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





