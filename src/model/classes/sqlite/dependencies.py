
from Chess_Model.src.model.classes.sqlite.database import SessionLocal
from Chess_Model.src.model.classes.sqlite.models import GamePositions
from Chess_Model.src.model.config.config import Settings
from sqlalchemy.orm import Session

from sqlalchemy import or_, and_, func
import chess

n_half_moves = Settings().halfMoveBin


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def delete_all_game_positions(db: Session = next(get_db())):
    try:
        # Open a new session
        with db as session:
            # Delete all records in the GamePositions table
            session.query(GamePositions).delete()

            # Commit the changes to the database
            session.commit()

            print("All records in GamePositions have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()

def insert_board_into_db(victor: str,board: chess.Board,db: Session = next(get_db())):
    game_info = board_to_GamePostition(board=board, victor=victor)
    gamesInDB = find_game(game_info=game_info, db=db)

    if gamesInDB is None:
        db.add(game_info)
    else:
        update_game(game_info=gamesInDB, new_game_info=game_info)

    db.commit()
    db.refresh(game_info if gamesInDB is None else gamesInDB)
    return 0

def find_game(game_info: GamePositions,db: Session = get_db()):
    results = db.query(GamePositions).filter(
            and_(GamePositions.piece_positions == game_info.piece_positions,
                GamePositions.castling_rights == game_info.castling_rights,
                GamePositions.en_passant == game_info.en_passant,
                GamePositions.turn == game_info.turn,
                GamePositions.greater_than_n_half_moves == game_info.greater_than_n_half_moves,
                GamePositions.repeated_position == game_info.repeated_position)
            ).first()
    return results

def update_game(game_info: GamePositions,new_game_info: GamePositions):
    game_info.white_wins += new_game_info.white_wins
    game_info.black_wins += new_game_info.black_wins
    game_info.stalemates += new_game_info.stalemates
    return game_info

def get_game_info(game_info: GamePositions,db: Session = get_db()):
    game = find_game(game_info=game_info,db=db)
    white_wins = game.white_wins
    black_wins = game.black_wins
    stalemates = game.stalemates
    return white_wins,black_wins,stalemates

def board_to_GamePostition(board: chess.Board,victor: str = "NA"):
    fen = board.fen()
    fen_components = fen.split(" ")
    piece_positions = fen_components[0]
    turn = fen_components[1]
    castling_rights = fen_components[2]
    en_passant = valid_en_passant(board=board,en_passant=fen_components[3])
    half_move_clock = int(fen_components[4])
    half_move_bin =  1 if half_move_clock > n_half_moves else  0
    repeated_position = times_position_repeated(board=board)
    white_wins = 0
    black_wins = 0
    stalemates = 0
    if victor == 'w':
        white_wins = 1
    elif victor == 'b':
        black_wins = 1
    elif victor == 's':
        stalemates = 1

    game = GamePositions(
        piece_positions = piece_positions,
        castling_rights = castling_rights,
        en_passant = en_passant,
        turn = turn,
        greater_than_n_half_moves = half_move_bin,
        repeated_position = repeated_position,
        white_wins = white_wins,
        black_wins = black_wins,
        stalemates = stalemates
    )

    return game
    
def valid_en_passant(board: chess.Board,en_passant: str):
    if en_passant != '-':
        # There is a potential en passant target
        target_square = chess.parse_square(en_passant)

        # Iterate through all legal moves to find an en passant move
        for move in board.legal_moves:
            if move.to_square == target_square and board.is_en_passant(move):
                return en_passant
    else:
        return en_passant
    
def times_position_repeated(board: chess.Board):
    rep = 0
    for i in range(0,5):
        if board.is_repetition(i):
            rep = i
    return rep

def fetch_all_game_positions(db: Session = next(get_db())):
    try:
        # Querying all records in GamePositions
        for game_position in db.query(GamePositions).yield_per(1):
            yield game_position
    finally:
        db.close()
        yield None

def get_row_count(db: Session = next(get_db())):
    try:
        # Counting the rows in the GamePositions table
        count = db.query(func.count(GamePositions.id)).scalar()
        return count
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        db.close()