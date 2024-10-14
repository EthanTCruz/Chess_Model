from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
import json
import hashlib
from chess_engine.src.model.config.config import settings

Base = declarative_base()

def get_hash(piece_positions,castling_rights,en_passant,turn):
        game_string = (
            piece_positions
            + castling_rights
            + en_passant
            + turn
            
        )
        hash_object = hashlib.sha256(game_string.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig


class GamePositions(Base):
    __tablename__ = "GamePositions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    piece_positions = Column(String, index=True)
    castling_rights = Column(String, index=True)
    en_passant = Column(String, index=True)
    turn = Column(String, index=True)

    white_wins = Column(Integer)
    black_wins = Column(Integer)
    stalemates = Column(Integer)

    fen = f"{piece_positions} {turn} {castling_rights} {en_passant} {0} {1}"

    

    @property
    def get_hash(self):
        game_string = (
            self.piece_positions
            + self.castling_rights
            + self.en_passant
            + self.turn
            
        )
        hash_object = hashlib.sha256(game_string.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

    @property
    def total_wins(self):
        total_wins = self.white_wins + self.black_wins + self.stalemates
        return total_wins

    @property
    def win_buckets(self):
        total_wins = self.total_wins
        if total_wins > 0:
            mean_w = self.white_wins / total_wins
            mean_b = self.black_wins / total_wins
            mean_s = self.stalemates / total_wins
            return [mean_w, mean_b, mean_s]
        else:
            return [0, 0, 0]

    @staticmethod
    def from_json(json_str, win_buckets):
        data = json.loads(json_str)
        win_buckets = json.loads(win_buckets)
        return GamePositions(
            piece_positions=data["piece_positions"],
            castling_rights=data["castling_rights"],
            en_passant=data["en_passant"],
            turn=data["turn"],
            white_wins=win_buckets["white_wins"],
            black_wins=win_buckets["black_wins"],
            stalemates=win_buckets["stalemates"]
        )

class GamePositionRollup(GamePositions):
    __tablename__ = "GamePositionRollup"
    id = Column(Integer, ForeignKey("GamePositions.id"), primary_key=True)
    game_position = relationship("GamePositions", backref="rollup_position")

# Using joined table inheritance
class TrainGamePositions(GamePositions):
    __tablename__ = "TrainGamePositions"
    id = Column(Integer, ForeignKey("GamePositions.id"), primary_key=True)
    game_position = relationship("GamePositions", backref="train_position")

class TestGamePositions(GamePositions):
    __tablename__ = "TestGamePositions"
    id = Column(Integer, ForeignKey("GamePositions.id"), primary_key=True)
    game_position = relationship("GamePositions", backref="test_position")

class ValidationGamePositions(GamePositions):
    __tablename__ = "ValidationGamePositions"
    id = Column(Integer, ForeignKey("GamePositions.id"), primary_key=True)
    game_position = relationship("GamePositions", backref="validation_position")


