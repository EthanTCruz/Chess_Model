from pydantic_settings import BaseSettings

class Settings(BaseSettings, case_sensitive=True):
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_score_db: int = 1
    
    nnGenBatchSize: int = 8192
    nnBatchSize: int = 2048
    nnEpochs: int = 192
    nnTestSize: float = 0.15
    nnValidationSize: float  = 0.15
    

    srcModelDirectory: str = './Chess_Model/src/model'
    ModelFilePath: str =f"{srcModelDirectory}/chess_model/"
    ModelFilename: str ="model.h5"
    scores_file: str = f"{srcModelDirectory}/data/data.csv"
    pgn_file: str = f"{srcModelDirectory}/pgn/Adams.pgn"
    games_csv_file: str = f"{srcModelDirectory}/data/games.csv"
    predictions_board: str = f"{srcModelDirectory}/data/predictions.csv"
    self_play: str = f"{srcModelDirectory}/data/self_play.pgn"
    persist_model: bool = True
    scaler_weights: str = f"{srcModelDirectory}/data/scaler.joblib"
    SelfPlayModelFilename: str ="self_play_model"
    samplePgn: str = f"{srcModelDirectory}/pgn/sample.pgn"
    nnPredictionsCSV: str = f"{srcModelDirectory}/data/nn_predictions.csv"
    selfTrainBaseMoves: str = f"{srcModelDirectory}/data/simGames.csv"

    #should run under assumption score depth will always be greater than mate depth
    score_depth: int = 1
    player: str = 'w'
    endgame_table: str = f"{srcModelDirectory}/data/EndgameTbl/"
    minimumEndgamePieces: int = 5
    trainModel: bool = True
    selfTrain: bool = True

    copy_file: str = f"{srcModelDirectory}/data/copy_data.csv"
    trainingFile: str = f"{srcModelDirectory}/data/training.csv"
    testingFile: str = f"{srcModelDirectory}/data/testing.csv"
    validationFile: str = f"{srcModelDirectory}/data/validation.csv"
    evalModeFile: str = f"{srcModelDirectory}/data/modelEval.csv"

    GOOGLE_APPLICATION_CREDENTIALS: str = "C:\\Users\\ethan\\git\\Full_Chess_App\\Chess_Model\\terraform\\secret.json"
    BUCKET_NAME: str = "chess-model-weights"
    class Config:
        env_prefix = ''

