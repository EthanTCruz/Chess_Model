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

from Chess_Model.src.model.classes.endgame import endgamePicker
from Chess_Model.src.model.classes.mongo_functions import mongo_data_pipe
from Chess_Model.src.model.classes.torch_model import model_operator
from Chess_Model.src.model.classes.torch_movepicker import move_picker

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

# nn = cnn_model(**nn_kwargs)

# mp = cnn_move_picker(neuralNet=nn)


mdp = mongo_data_pipe()



def main():

    

    # cowsay.cow(f"Converting pgn file to sqlite db")    
    # pgn_to_db(pgn_file=pgn_file)
    # cowsay.cow(f"populating mongodb")    
    # initialize_collections()
    # cowsay.cow(f"testing model functions")    
    # test_pt_model()
    board = chess.Board()
    eval = use_model(board=board)
    print(eval)


    return 0

def verify_functionality_on_sample_dataset():
    cowsay.cow(f"Converting pgn file to sqlite db")    
    pgn_to_db(pgn_file=pgn_file)
    cowsay.cow(f"populating mongodb")    
    initialize_collections()
    cowsay.cow(f"testing model functions")    
    test_pt_model()
    board = chess.Board()
    use_model(board=board)
    


def use_model(board: chess.Board = chess.Board()):
    mp = move_picker()
    move = mp.use_model(board=board)

    return move

def initialize_collections():
    mdp.open_connections()
    
    mdp.initialize_data(batch_size=512)
    mdp.close_connections()


def eval_board(board: chess.Board):
    return mp.use_model(board=board)

def test_pt_model():
    # initialize_collections()
    model = model_operator()
    model.Create_and_Train_Model(num_workers = 0,num_epochs=16,save_model=False)
    # model.load_and_evaluate_model(model_path=s.torch_model_file)


def pgn_to_db(pgn_file = pgn_file,db: Session = SessionLocal()):

    delete_all_game_positions(db = db)
    pgn_obj = pgn_processor(pgn_file=pgn_file)
    pgn_obj.pgn_fen_to_sqlite()

    
    return 0 


def test_endgame(board:chess.Board):

    ep = endgamePicker()
    results = ep.find_endgame_best_move(board=board)
    print(results)
    return results



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



def create_csv():
    # Check if the file exists and remove it
    if os.path.exists(eval_file):
        os.remove(eval_file)

    # Create a new CSV file with the column headers
    with open(eval_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        columns = ["epochs","loss","accuracy"]
        writer.writerow(columns)




if __name__ == "__main__":
    main()


