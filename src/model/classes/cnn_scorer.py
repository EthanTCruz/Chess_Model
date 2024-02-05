import chess
from Chess_Model.src.model.classes.endgame import endgamePicker
import numpy as np
from math import ceil
from Chess_Model.src.model.classes.sqlite.dependencies import board_to_GamePostition
from Chess_Model.src.model.classes.sqlite.models import GamePositions

class boardCnnEval:
    def __init__(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = None):
        self.setup_parameters(fen=fen,board=board)
        
        self.ep = endgamePicker()
        
        self.endgameAmount = 5

    def setup_parameters(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = None):
        self.fen = fen

        self.board = chess.Board(fen=fen)
        if board is not None:
            self.board = board
            self.fen = board.fen()
        self.fen_components = fen.split(" ") 
    


    def setup_parameters_gamepositions(self,game: GamePositions):
        self.game = game  
        self.board = chess.Board(self.game.piece_positions)
        if self.game.turn == 'b':
            self.board.turn = chess.BLACK
            self.board.halfmove_clock = 2
            self.board.fullmove_number = 1


    def get_features(self):
        scores = self.get_board_scores()
        features = ["moves(id)"] + list(scores.keys())
        return features

    def get_metadata(self):
        dict_results = {}

        game_results = []
        turn = self.game.turn

        white_turn = 1 if turn == 'w' else 0
        black_turn = 1 if turn == 'b' else 0

        dict_results["white turn"] = white_turn
        dict_results["black turn"] = black_turn

        game_results.append(white_turn)
        game_results.append(black_turn)

        white_queenside = 0
        black_queenside = 0
        white_kingside = 0
        black_kingside = 0

        for rights in self.game.castling_rights:
            if rights == 'q':
                black_queenside = 1
            elif rights == 'Q':
                white_queenside = 1
            elif rights == 'k':
                black_kingside = 1
            elif rights == 'K':
                white_kingside = 1

        dict_results["white queenside castle"] = white_queenside
        dict_results["white kingside castle"] = white_queenside
        dict_results["black queenside castle"] = black_kingside
        dict_results["black kingside castle"] = black_kingside

        is_endgame = self.is_endgame()
        if is_endgame == 1:
            endgame_scores = self.endgame_status()
        else:
            endgame_scores = [0,0]

        dict_results["white wdl"] = endgame_scores[0]
        dict_results["black wdl"] = endgame_scores[1]


        game_results.append(white_queenside)
        game_results.append(white_kingside)
        game_results.append(black_queenside)
        game_results.append(black_kingside)                        
        game_results.append(self.game.greater_than_n_half_moves)



        
        return dict_results
    
    def is_endgame(self):

        count = self.ep.count_pieces(board=self.board)

        if count <= self.endgameAmount:
            return 1
        else:
            return 0

    def endgame_status(self):
        w_or_b = [0,0]

        results = self.ep.endgame_status(board=self.board)
        if results > 0:
            if self.board.turn:
                w_or_b[0] = 1
            else:
                w_or_b[1] = 1

        elif results < 0:
            if self.board.turn:
                w_or_b[1] = 1
            else:
                w_or_b[0] = 1

        return w_or_b
    def open_tables(self):
        self.ep.open_tables()

    def close_tables(self):
        self.ep.close_tables()

    def get_game_results(self):
        results = []
        dict_results = {}
        means = self.game.win_buckets
        results += means
        dict_results["white mean"] = means[0]
        dict_results["black mean"] = means[1]
        dict_results["stalemate mean"] = means[2]
        return dict_results

    def get_board_scores(self,victor="NA"):
        dict_results = {}
        results = []
        metadata = self.get_metadata()
        game_results = self.get_game_results()
        board = chess.Board(self.game.piece_positions)
        piece_locations = self.get_all_piece_locations(original_board=board)
        dict_results.update(piece_locations)
        dict_results.update(metadata)
        dict_results.update(game_results)
        # results += piece_locations
        white_attacks,black_attacks = self.w_b_attacks()
        # results.append(white_attacks)
        # results.append(black_attacks)
        white_advantage = np.where(white_attacks > black_attacks, 1, 0)
        black_advantage = np.where(black_attacks > white_attacks, 1, 0)
        # results.append(white_advantage)
        # results.append(black_advantage)
        dict_results["white advantage positions"] = white_advantage.flatten()
        dict_results["black advantage positions"] = black_advantage.flatten()
        return dict_results
    
    def piece_positons(self,white: bool,piece: str):
        if white:
            piece = piece.upper()
        else:
            piece = piece.lower()
        
    def get_all_piece_locations(self,original_board: chess.Board):
        board = original_board.copy()
        dict_results = {}
        black_pieces = [chess.Piece.from_symbol('n'),chess.Piece.from_symbol('b'),chess.Piece.from_symbol('r'),chess.Piece.from_symbol('q'),chess.Piece.from_symbol('k'),chess.Piece.from_symbol('p')]
        white_pieces = [chess.Piece.from_symbol('N'),chess.Piece.from_symbol('B'),chess.Piece.from_symbol('R'),chess.Piece.from_symbol('Q'),chess.Piece.from_symbol('K'),chess.Piece.from_symbol('P')]
        piece_map = board.piece_map()
        white_results = self.pieces_to_matrix(pieces=white_pieces,piece_map=piece_map)
        black_results = self.pieces_to_matrix(pieces=black_pieces,piece_map=piece_map)

        dict_results["white knight positions"] = white_results[0].flatten()
        dict_results["white bishop positions"] = white_results[1].flatten()
        dict_results["white rook positions"] = white_results[2].flatten()
        dict_results["white queen positions"] = white_results[3].flatten()
        dict_results["white king positions"] = white_results[4].flatten()
        dict_results["white pawn positions"] = white_results[5].flatten()

        dict_results["black knight positions"] = black_results[0].flatten()
        dict_results["black bishop positions"] = black_results[1].flatten()
        dict_results["black rook positions"] = black_results[2].flatten()
        dict_results["black queen positions"] = black_results[3].flatten()
        dict_results["black king positions"] = black_results[4].flatten()
        dict_results["black pawn positions"] = black_results[5].flatten()

        return dict_results
        

    def pieces_to_matrix(self,pieces, piece_map):
        zeros = np.zeros((8,8),dtype=int)
        results = []
        for _ in range(6):
            results.append(np.copy(zeros))
        for i in range(0,len(pieces)):
            current_piece = pieces[i]
            current_results = results[i]
            keys = list(piece_map.keys())

            for key in keys:
                if piece_map[key] == current_piece:
                    row,col = sequence_to_rc(seq=key)
                    current_results[row][col] += 1
                    piece_map.pop(key)
        return results
    
    def w_b_attacks(self):
        white_attacks = self.calculate_square_attacks(white=True)
        black_attacks = self.calculate_square_attacks(white=False)
        return white_attacks, black_attacks
    
    def calculate_square_attacks(self,white:bool):
        attack_matrix = np.zeros((8, 8),dtype=int)
        pro = chess.WHITE if white else chess.BLACK
        # Iterate over all squares
        for square in chess.SQUARES:
            attackers = self.board.attackers(pro, square)
            count_attackers = str(attackers).count("1")

            # Convert the square index to row and column for the matrix
            row, col = divmod(square, 8)
            attack_matrix[row, col] = count_attackers

        return attack_matrix

    def get_features(self):
        board = chess.Board()
        game = board_to_GamePostition(board=board,victor='w')
        self.setup_parameters_gamepositions(game=game)
        scores = self.get_board_scores()
        features =  list(scores.keys())
        return features
    
def sequence_to_rc(seq: int):
        row = 7 - (int(ceil((seq + 1)/8)) - 1) 
        col = (seq+1) % 8 - 1
        return row,col
