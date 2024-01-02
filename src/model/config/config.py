from pydantic_settings import BaseSettings

class Settings(BaseSettings, case_sensitive=True):
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_score_db: int = 1

    srcModelDirectory: str = 'C:/Users/ethan/git/chess_model/src/model'
    ModelFilePath: str =f"{srcModelDirectory}/"
    ModelFilename: str ="chess_model"
    scores_file: str = f"{srcModelDirectory}/data/data.csv"
    pgn_file: str = f"{srcModelDirectory}/pgn/Adams.pgn"
    games_csv_file: str = f"{srcModelDirectory}/data/games.csv"
    predictions_board: str = f"{srcModelDirectory}/data/predictions.csv"
    persist_model: bool = True

    #should run under assumption score depth will always be greater than mate depth
    score_depth: int = 1
