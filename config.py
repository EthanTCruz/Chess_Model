from pydantic_settings import BaseSettings

class Settings(BaseSettings, case_sensitive=True):
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_score_db: int = 1
    redis_mate_db: int = 2
    ModelFilePath: str ="./"
    ModelFilename: str ="chess_model"
    scores_file: str = "./data/data.csv"
    pgn_file: str = "./pgn/Adams.pgn"
    games_csv_file: str = "./data/games.csv"
    predictions_board: str = './data/predictions.csv'
    persist_model: bool = True

    #should run under assumption score depth will always be greater than mate depth
    score_depth: int = 2
    mate_depth: int = 3