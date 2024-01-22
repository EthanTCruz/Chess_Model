import sys
import os
import csv
sys.path.append('./')


from  Chess_Model.src.model.classes.game_analyzer import game_analyzer
import chess
from Chess_Model.src.model.classes.pgn_processor import pgn_processor
from Chess_Model.src.model.classes.nn_model import neural_net
import cowsay
from Chess_Model.src.model.classes.move_picker import move_picker
from Chess_Model.src.model.config.config import Settings
import time
from Chess_Model.src.model.classes.model_trainer import trainer
from Chess_Model.src.model.classes.endgame import endgamePicker



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
#pgn_file=s.samplePgn


nn = neural_net(filename=scores_file,target_feature='w/b',
                test_size=test_size,ModelFilename = ModelFilename,
                ModelFilePath=ModelFilePath,player='w',
                predictions_board=predictions_board,epochs=epochs,
                trainModel=s.trainModel,batch_size=batch_size)



mp = move_picker(neuralNet=nn)





def main():
    #test()
    tune_parameters()
    # if s.trainModel:
    #     train_and_test_model()
    # if s.selfTrain:
    #     test_self_train()
    #board = chess.Board(fen='8/8/6K1/8/2k5/2P2R2/1P6/8 w - - 0 44')
    #test_endgame(board=board)
    #train_and_test_model()

    #highest_scoring_move()
    return 0

def highest_scoring_move():
    #board = chess.Board(fen='r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4')
    board = chess.Board()
    # board.push_san("e4")
    # board.push_san("e5")
    # board.push_san("Bc4")
    # board.push_san("Nc6")
    # board.push_san("Qh5")
    #sample = 'r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3'
    #board.push_san("Nf6")

    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nc3")
    board.push_san("Bc5")
    board.push_san("Nb1")
    #board.push_san("Qh4")
    #board.push_san("Nc3")
    #board.push_san("Qxf2")

    start_time = time.time()
    move = mp.use_model(board=board)

    end_time = time.time()
    duration = end_time - start_time 
    print(f"Model took {duration} seconds to run., Move is: {move}, should be h4f2")

def test_scholar_mate():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")
    board.push_san("Nc6")
    board.push_san("Qh5")
    board.push_san("Nf6")
    '''
    for key in self.r_mate.keys("*'c4d3', 'c6b4'*"):
    print(f"{key} :  {self.r_mate.get(key)}")
    '''
    #['c4d3', 'c6b4'] :  1 error
    mp.use_model(board=board)

#def test_move_picker():



def train_and_test_model():

    # if os.path.exists(scores_file):
    #     os.remove(scores_file)
    # if os.path.exists(games_csv_file):
    #     os.remove(games_csv_file)





    # pgn_obj = pgn_processor(pgn_file=pgn_file,csv_file=games_csv_file)
    # cowsay.cow(f"Converting pgn file to csv: {games_csv_file}")    
    # pgn_obj.pgn_fen_to_csv()
    # del pgn_obj
    # cowsay.cow(f"Generating feature data from pgn boards in csv: {scores_file}")
    # gam_an_obj = game_analyzer(scores_file=scores_file)
    # gam_an_obj.process_csv_boards(csv_file=games_csv_file)
    # del gam_an_obj
    
    cowsay.cow("Create neural net")
    
    cowsay.cow(f"Training neural net on {scores_file} and saving weights to {ModelFilename}")
    nn.create_and_evaluate_model_batch()
    
    #nn.score_board(board_key="['g1h3', 'h7h5', 'h3g1']:rnbqkbnr/ppppppp1/8/7p/8/8/PPPPPPPP/RNBQKBNR b KQkq - 1 2")
    return 0

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
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")
    board.push_san("Nc6")
    board.push_san("Qh5")
    board.push_san("Nf6")
    gam_an_obj = game_analyzer(scores_file=scores_file)
    gam_an_obj.process_single_fen(fen=board.fen())



def get_sample_board():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")
    board.push_san("Nc6")
    board.push_san("Qh5")
    board = chess.Board()
    #board.push_san("Nf6")
    return board

def tune_parameters():
    create_csv()
    epoch_sizes = [256,512]
    batch_sizes = [2048]
    for b in batch_sizes:
        for e in epoch_sizes:
            nn = neural_net(filename=scores_file,target_feature='w/b',
                test_size=test_size,ModelFilename = ModelFilename,
                ModelFilePath=ModelFilePath,player='w',
                predictions_board=predictions_board,epochs=e,
                trainModel=s.trainModel,batch_size=b)
            loss,accuracy = nn.create_and_evaluate_model_batch()
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


