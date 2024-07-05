import chess
from Chess_Model.src.model.classes.endgame import endgamePicker
import numpy as np
from math import ceil
from Chess_Model.src.model.classes.sqlite.dependencies import board_to_GamePostition
from Chess_Model.src.model.classes.sqlite.models import GamePositions
from Chess_Model.src.model.config.config import Settings
from Chess_Model.src.model.classes.metadata_scorer import metaDataBoardEval


class boardCnnEval:
    def __init__(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = None):
        self.half_move_amount = Settings().halfMoveBin
        self.setup_parameters(fen=fen,board=board)
        
        self.ep = endgamePicker()
        
        self.endgameAmount = 5

        self.zeros_matrix = np.zeros((8,8),dtype=int)
        self.black_pieces = [chess.Piece.from_symbol('n'),chess.Piece.from_symbol('b'),chess.Piece.from_symbol('r'),chess.Piece.from_symbol('q'),chess.Piece.from_symbol('k'),chess.Piece.from_symbol('p')]
        self.white_pieces = [chess.Piece.from_symbol('N'),chess.Piece.from_symbol('B'),chess.Piece.from_symbol('R'),chess.Piece.from_symbol('Q'),chess.Piece.from_symbol('K'),chess.Piece.from_symbol('P')]
        self.all_pieces = self.white_pieces + self.black_pieces


    def setup_parameters(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = None):
        self.fen = fen

        self.board = chess.Board(fen=fen)
        if board is not None:
            self.board = board
            self.fen = board.fen()
        self.fen_components = fen.split(" ") 
        return 0


    def setup_parameters_gamepositions(self,game: GamePositions):
        self.game = game  
        #have to reconstruct full fen instead of just piece_positions
        if game.greater_than_n_half_moves == 1:
            half_moves = self.half_move_amount
            full_moves = 2 * half_moves
        else:
            half_moves = 0
            full_moves = half_moves

        fen = f"{game.piece_positions} {game.turn} {game.castling_rights} {game.en_passant} {half_moves} {full_moves}"
        
        self.board = chess.Board(fen)

        return 0


    def get_features(self):
        scores = self.get_board_scores()
        features = ["moves(id)"] + list(scores.keys())
        return features

    def get_metadata(self):
        metaDataEvaluator = metaDataBoardEval(game=self.game)
        
        dict_results = metaDataEvaluator.get_board_scores()

        turn = self.game.turn

        white_turn = 1 if turn == 'w' else 0
        black_turn = 1 if turn == 'b' else 0

        dict_results["white turn"] = white_turn
        dict_results["black turn"] = black_turn


 
        return dict_results
    
    def en_passant_board(self):
        zeros = self.zeros_matrix.copy()
        if self.game.en_passant != '-':
        # There is a potential en passant target
            target_square = chess.parse_square(self.game.en_passant)
            row, col = divmod(target_square, 8)
            zeros[row, col] = 1
        return zeros


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


    
    
    def get_board_scores(self):
        dict_results = {}

        dict_results["metadata"] = list(self.get_metadata().values())



        dict_results["positions_data"] = board_to_bitboards(self.board)


        
        dict_results["game_results"] = list(self.get_game_results().values())

        return dict_results    



def board_to_bitboards(board):
    bitboards = []
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in chess.PIECE_TYPES:
            bitboard = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == piece_type and piece.color == color:
                    bitboard |= 1 << square
            # color_name = 'White' if color == chess.WHITE else 'Black'
            # piece_name = chess.piece_name(piece_type).capitalize()
            bitboards.append(bitboard)
    return bitboards