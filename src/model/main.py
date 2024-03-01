import sys
import os
import csv
import cowsay
import chess
import time

sys.path.append('./')
from sqlalchemy.orm import  Session
from Chess_Model.src.model.classes.sqlite.database import SessionLocal
from Chess_Model.src.model.classes.sqlite.dependencies import delete_all_game_positions
from Chess_Model.src.model.classes.pgn_processor import pgn_processor

from Chess_Model.src.model.config.config import Settings

from Chess_Model.src.model.classes.model_trainer import trainer
from Chess_Model.src.model.classes.endgame import endgamePicker


from Chess_Model.src.model.classes.cnn_dataGenerator import data_generator as cnn_data_generator
from Chess_Model.src.model.classes.cnn_game_analyzer import game_analyzer as cnn_game_analyzer
from Chess_Model.src.model.classes.cnn_model import convolutional_neural_net as cnn_model
from Chess_Model.src.model.classes.cnn_move_picker import move_picker as cnn_move_picker

s = Settings()
ModelFilePath=s.ModelFilePath
ModelFilename=s.ModelFilename
scores_file = s.scores_file
pgn_file = s.pgn_file
games_csv_file = s.games_csv_file
predictions_board = s.predictions_board

epochs = s.nnEpochs
batch_size = s.nnBatchSize
test_size = s.nnTestSize

persist_model = s.persist_model
score_depth = s.score_depth
eval_file = s.evalModeFile
if s.useSamplePgn:
    pgn_file=s.samplePgn

target_features = ["white mean","black mean","stalemate mean"]
nn_kwargs = {}
nn_kwargs["filename"]=scores_file
nn_kwargs["target_feature"]=target_features
nn_kwargs["test_size"]=test_size
nn_kwargs["ModelFilename"]=ModelFilename
nn_kwargs["ModelFilePath"]=ModelFilePath
nn_kwargs["player"]='w'
nn_kwargs["predictions_board"]=predictions_board
nn_kwargs["epochs"]=epochs
nn_kwargs["trainModel"]=s.trainModel
nn_kwargs["batch_size"]=batch_size
#nn = neural_net(**nn_kwargs)

nn = cnn_model(**nn_kwargs)

mp = cnn_move_picker(neuralNet=nn)





def main():
    #test()
    #test_process_fen()
    tune_parameters()

    # if s.trainModel:
    #     train_and_test_model()
    # if s.selfTrain:
    #     test_self_train()
    #board = chess.Board(fen='8/8/6K1/8/2k5/2P2R2/1P6/8 w - - 0 44')
    #test_endgame(board=board)
    #train_and_test_model()
    #pgn_to_db()

    #highest_scoring_move()

    return 0

def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # Uncomment the following line if you also want to remove subdirectories
                # shutil.rmtree(file_path)
                pass
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def pgn_to_db(db: Session = SessionLocal()):

    delete_all_game_positions(db = db)
    if os.path.exists(scores_file):
        os.remove(scores_file)
    if os.path.exists(games_csv_file):
        os.remove(games_csv_file)


    pgn_obj = pgn_processor(pgn_file=pgn_file,csv_file=games_csv_file)
    cowsay.cow(f"Converting pgn file to db: {games_csv_file}")    
    pgn_obj.pgn_fen_to_sqlite(db = db)
    del pgn_obj
    cowsay.cow(f"Generating feature data from pgn boards in csv: {scores_file}")
    gam_an_obj = cnn_game_analyzer(scores_file=scores_file)
    gam_an_obj.open_endgame_tables()
    gam_an_obj.process_sqlite_boards()
    gam_an_obj.close_endgame_tables()
    del gam_an_obj
    
    return 0 

def highest_scoring_move():
    nn = cnn_model(**nn_kwargs)
    mp = cnn_move_picker(neuralNet=nn)

    board = chess.Board()

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nc3")
    board.push_san("Bc5")
    board.push_san("Nb1")
    board.push_san("Qh4")
    #board.push_san("Nc3")
    #board.push_san("Qxf2")

    start_time = time.time()
    move = mp.use_model(board=board)

    end_time = time.time()
    duration = end_time - start_time 
    print(f"Model took {duration} seconds to run., Move is: {move}, should be h4f2")




def use_model(board: chess.Board = chess.Board()):

    move = mp.use_model(board=board)

    return move


def test_endgame(board:chess.Board):

    ep = endgamePicker()
    results = ep.find_endgame_best_move(board=board)
    print(results)
    return results


def test_self_train():
    t = trainer(filename=scores_file,target_feature='w/b',
                test_size=0.3,ModelFilename = ModelFilename,
                ModelFilePath=ModelFilePath,player='w',
                predictions_board=predictions_board,epochs=100,
                neuralNet = nn)
    t.self_train(iterations=1,depth=2)

def test_process_fen():
    board = get_sample_board()
    gam_an_obj = cnn_game_analyzer(scores_file=scores_file)
    info =  gam_an_obj.process_single_board(board=board)
    return info



def get_sample_board():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")
    board.push_san("Nc6")
    board.push_san("Qh5")
    board.push_san("Nf6")
    #board = chess.Board()

    return board

def tune_parameters():
    target_features = ["white mean","black mean","stalemate mean"]
    create_csv()
    delete_files_in_directory(directory=s.nnLogDir)
    if not s.trainDataExists:
        pgn_to_db()
        dg = cnn_data_generator(filename=scores_file,target_feature=target_features,
                    test_size=test_size,ModelFilename = ModelFilename,
                    ModelFilePath=ModelFilePath,player='w',
                    predictions_board=predictions_board,
                    trainModel=s.trainModel)
        dg.initialize_datasets()
        del dg

    epoch_sizes = [16,32,64,128]
    batch_sizes = [128,512,1024,2048,5012]
    for b in batch_sizes:
        for e in epoch_sizes:

            nn = cnn_model(filename=scores_file,target_feature=target_features,
                test_size=test_size,ModelFilename = ModelFilename,
                ModelFilePath=ModelFilePath,player='w',
                predictions_board=predictions_board,epochs=e,
                trainModel=s.trainModel,batch_size=b)
            
            loss,accuracy = nn.create_and_evaluate_model()

            with open(eval_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = [e,b,loss,accuracy]
                writer.writerow(row)
    return 0

def create_csv():
    # Check if the file exists and remove it
    if os.path.exists(eval_file):
        os.remove(eval_file)

    # Create a new CSV file with the column headers
    with open(eval_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        columns = ["epochs","batch_size","loss","accuracy"]
        writer.writerow(columns)


if __name__ == "__main__":
    main()


