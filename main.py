from redis_populator import populator
import game_analyzer
import chess
from pgn_processor import pgn_processor
from nn_model import neural_net
import os
import cowsay

scores_file = "data.csv"
pgn_file = "Adams.pgn"
games_csv_file = "games.csv"

def main():
    cowsay.cow("Start")

    if os.path.exists(scores_file):
        os.remove(scores_file)
    if os.path.exists(games_csv_file):
        os.remove(games_csv_file)

    gam_an_obj = game_analyzer.game_analyzer(player='w',output_file=scores_file)
    pgn_obj = pgn_processor(pgn_file=pgn_file,csv_file=games_csv_file)
        
    pgn_obj.pgn_fen_to_csv()
    cowsay.cow("Converted pgn to csv")
    gam_an_obj.process_csv_boards(csv_file=games_csv_file)
    cowsay.cow("Evaluated csv boards")

    cowsay.cow("Create neural net")
    nn = neural_net(filename=scores_file,target_feature='w/b',test_size=0.3)
    nn.create_and_evaluate_model()
    
    '''
    nn = neural_net(filename=scores_file, target_feature='w/b', test_size=0.3)
    nn.create_and_evaluate_model()


    board = chess.Board()
    red_obj = redis_populator.populator(depth=2,board=board)
    red_obj.reset_and_fill_redis()
    gam_an_obj.process_redis_boards()
    
    
    '''
    print("Finish")



def test_model():
    return 0
def train_model():
    return 0

if __name__ == "__main__":
    main()