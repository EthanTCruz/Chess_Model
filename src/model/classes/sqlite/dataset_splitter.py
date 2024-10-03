
from chess_engine.src.model.classes.sqlite.dependencies import get_db
from chess_engine.src.model.classes.sqlite.models import (GamePositions,
                                                          GamePositionRollup,
                                                          TrainGamePositions,
                                                          ValidationGamePositions,
                                                          TestGamePositions)
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.orm import load_only
import random

def split_game_positions_in_batches(train_pct: float, 
                                    test_pct: float,
                                    validation_pct: float,
                                    batch_size: int = 1000,
                                    db: Session = next(get_db())):
    # Validate the input percentages add up to 1.0
    if not (train_pct + test_pct + validation_pct == 1.0):
        raise ValueError("The sum of train, test, and validation percentages must be 1.0")

    offset = 0
    while True:
        # Retrieve a batch of records from GamePositions
        game_positions_batch = db.query(GamePositions).options(load_only(
            "id", "piece_positions", "castling_rights", "en_passant", "turn", "greater_than_n_half_moves",
            "white_wins", "black_wins", "stalemates"
        )).offset(offset).limit(batch_size).all()

        if not game_positions_batch:
            # No more records to process
            break

        # Shuffle records for random distribution
        random.shuffle(game_positions_batch)

        # Determine the sizes of each set in the current batch
        total_records = len(game_positions_batch)
        train_size = int(total_records * train_pct)
        test_size = int(total_records * test_pct)
        validation_size = total_records - train_size - test_size

        # Split the data
        train_records = game_positions_batch[:train_size]
        test_records = game_positions_batch[train_size:train_size + test_size]
        validation_records = game_positions_batch[train_size + test_size:]

        # Insert the records into respective tables
        for record in train_records:
            train_record = TrainGamePositions(**{column.name: getattr(record, column.name) for column in GamePositions.__table__.columns})
            db.add(train_record)

        for record in test_records:
            test_record = TestGamePositions(**{column.name: getattr(record, column.name) for column in GamePositions.__table__.columns})
            db.add(test_record)

        for record in validation_records:
            validation_record = ValidationGamePositions(**{column.name: getattr(record, column.name) for column in GamePositions.__table__.columns})
            db.add(validation_record)

        # Commit the changes to the database after processing the batch
        db.commit()

        # Update the offset to get the next batch
        offset += batch_size


