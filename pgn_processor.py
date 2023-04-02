import chess.pgn



class pgn_trainer():
    def __init__(self,pgn_file) -> None:
        self.pgn_file = pgn_file

pgn = open('Adams.pgn')



my_list = []

while True:
    game = chess.pgn.read_game(pgn)
    if game is None:
        break  # end of file
    board = game.board()
    victor = 's'
    if game.headers["Result"] == '1-0':
        victor = 'w'
    elif game.headers["Result"] == '1-0':
        victor = 'b'
    for move in game.mainline_moves():
        board.push(move=move)

        print(board.fen())
    my_list.append(game)
