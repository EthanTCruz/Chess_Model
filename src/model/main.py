import sys
sys.path.append('C:\\Users\\ethan\\git\\chess_model')

from src.model.classes.redis_populator import populator
from  src.model.classes.game_analyzer import game_analyzer
import chess
from src.model.classes.pgn_processor import pgn_processor
from src.model.classes.nn_model import neural_net
import os
import cowsay
from src.model.classes.move_picker import move_picker
import redis
import hiredis
from src.model.config.config import Settings
s = Settings()
ModelFilePath=s.ModelFilePath
ModelFilename=s.ModelFilename
scores_file = s.scores_file
pgn_file = s.pgn_file
games_csv_file = s.games_csv_file
predictions_board = s.predictions_board
redis_score_db = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_score_db))

persist_model = s.persist_model
score_depth = s.score_depth


nn = neural_net(filename=scores_file,target_feature='w/b',
                test_size=0.3,ModelFilename = ModelFilename,
                ModelFilePath=ModelFilePath,player='w',
                predictions_board=predictions_board,epochs=100,
                redis_score_db=redis_score_db)



mp = move_picker(redis_score_db=redis_score_db,
                 player='w')





def main():
    highest_scoring_move()
    #test_scholar_mate()
    #train_and_test_model()
    '''
    if not (persist_model):
        train_and_test_model()
    use_model()
    '''

def highest_scoring_move():
    board = chess.Board(fen='r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4')
    # board.push_san("e4")
    # board.push_san("e5")
    # board.push_san("Bc4")
    # board.push_san("Nc6")
    # board.push_san("Qh5")
    # board.push_san("Nf6")

    use_model(board=board)

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
    use_model(board=board)

#def test_move_picker():



def train_and_test_model():

    if os.path.exists(scores_file):
        os.remove(scores_file)
    if os.path.exists(games_csv_file):
        os.remove(games_csv_file)

    ga = {}
    player = 'w'

    gam_an_obj = game_analyzer(player=player,scores_file=scores_file)
    pgn_obj = pgn_processor(pgn_file=pgn_file,csv_file=games_csv_file)

    cowsay.cow(f"Converting pgn file to csv: {games_csv_file}")    
    pgn_obj.pgn_fen_to_csv()
    cowsay.cow(f"Generating feature data from pgn boards in csv: {scores_file}")
    gam_an_obj.process_csv_boards(csv_file=games_csv_file)

    
    cowsay.cow("Create neural net")
    
    cowsay.cow(f"Training neural net on {scores_file} and saving weights to {ModelFilename}")
    nn.create_and_evaluate_model()
    
    #nn.score_board(board_key="['g1h3', 'h7h5', 'h3g1']:rnbqkbnr/ppppppp1/8/7p/8/8/PPPPPPPP/RNBQKBNR b KQkq - 1 2")
    return 0

def use_model(board: chess.Board = chess.Board()):
    cowsay.cow("processing redis boards")
    red_obj = populator(depth=score_depth,board=board,
                        redis_score_db=redis_score_db)
    red_obj.reset_and_fill_redis()
    cowsay.cow("making predictions")
    nn.send_move_scores_to_redis(board)

    cowsay.cow("choosing move")

    move = mp.highest_average_move(board=board)
    redis_score_db.flushall()
    print(move)
    print("Finish")

    return 0





if __name__ == "__main__":
    main()