from pydantic_settings import BaseSettings
from math import sqrt
class Settings(BaseSettings): 
    nnGenBatchSize: int = 1
    nnScalarBatchSize: int = 10
    nnBatchSize: int = 100
    nnEpochs: int = 100
    nnTestSize: float = 0.15
    nnValidationSize: float  = 0.15

    srcModelDirectory: str = './Chess_Model/src/model'
    ModelFilePath: str =f"{srcModelDirectory}/chess_model/"
    ModelFilename: str = "model.h5"
    scores_file: str = f"{srcModelDirectory}/data/data.csv"
    pgn_file: str = f"{srcModelDirectory}/pgn/full_dataset/"
    games_csv_file: str = f"{srcModelDirectory}/data/games.csv"
    predictions_board: str = f"{srcModelDirectory}/data/predictions.csv"
    self_play: str = f"{srcModelDirectory}/data/self_play.pgn"
    persist_model: bool = True
    scaler_weights: str = f"{srcModelDirectory}/data/scaler.joblib"
    recordsData: str = f"{srcModelDirectory}/data/feature_data.tfrecord"
    recordsDataCopy: str = f"{srcModelDirectory}/data/feature_data_copy.tfrecord"
    recordsDataTrain: str = f"{srcModelDirectory}/data/train_data.tfrecord"
    recordsDataTest: str = f"{srcModelDirectory}/data/test_data.tfrecord"
    recordsDataValidation: str = f"{srcModelDirectory}/data/validation_data.tfrecord"

    SelfPlayModelFilename: str ="self_play_model"
    
    samplePgn: str = f"{srcModelDirectory}/pgn/sample_dataset/"
    nnPredictionsCSV: str = f"{srcModelDirectory}/data/nn_predictions.csv"
    selfTrainBaseMoves: str = f"{srcModelDirectory}/data/simGames.csv"
    nnLogDir: str = "./Chess_Model/logs/"
    nnModelCheckpoint: str = f"{ModelFilePath}checkpoints/"

    #should run under assumption score depth will always be greater than mate depth
    score_depth: int = 1
    player: str = 'w'
    endgame_table: str = f"{srcModelDirectory}/data/EndgameTbl/"
    minimumEndgamePieces: int = 5
    trainModel: bool = False
    selfTrain: bool = False
    trainDataExists: bool = True
    useSamplePgn: bool = True
    saveToBucket: bool = False
    tuneParameters: bool = False

    #MCST parameters:
    UCB_Constant: float = 0.1

    copy_file: str = f"{srcModelDirectory}/data/copy_data.csv"
    trainingFile: str = f"{srcModelDirectory}/data/training.csv"
    testingFile: str = f"{srcModelDirectory}/data/testing.csv"
    validationFile: str = f"{srcModelDirectory}/data/validation.csv"
    evalModeFile: str = f"{srcModelDirectory}/data/modelEval.csv"

    GOOGLE_APPLICATION_CREDENTIALS: str = "C:\\Users\\ethan\\git\\Full_Chess_App\\Chess_Model\\terraform\\secret.json"
    BUCKET_NAME: str = "chess-model-weights"
    matrixScalerFile: str = f"{srcModelDirectory}/data/matrixScaler.joblib"

    # Mongo settings

    validation_collection_key: str = "validation_data"
    testing_collection_key: str = "testing_data"
    training_collection_key: str = "training_data"
    main_collection_key: str = "main_collection"

    db_name: str = "mydatabase"
    metadata_key: str = 'metadata'
    bitboards_key: str = 'positions_data'
    results_key: str = 'game_results'

    mongo_host: str = "localhost"
    mongo_port: int = 27017

    mongo_url: str = f"mongodb://{mongo_host}:{mongo_port}/"
    
    num_workers: int = 0

    halfMoveBin: int = 25
    
    class Config:
        env_prefix = ''

