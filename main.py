from redis_populator import populator
import game_analyzer
import chess
from pgn_processor import pgn_processor
from nn_model import neural_net
import os
import cowsay
from move_picker import move_picker
import redis
import hiredis
from config import Settings
s = Settings()
ModelFilePath=s.ModelFilePath
ModelFilename=s.ModelFilename
scores_file = s.scores_file
pgn_file = s.pgn_file
games_csv_file = s.games_csv_file
predictions_board = s.predictions_board
redis_score_db = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_score_db))
redis_mate_db = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_mate_db))
persist_model = s.persist_model
score_depth = s.score_depth
mate_depth = s.mate_depth

nn = neural_net(filename=scores_file,target_feature='w/b',
                test_size=0.3,ModelFilename = ModelFilename,
                ModelFilePath=ModelFilePath,player='w',
                predictions_board=predictions_board,epochs=100,
                redis_score_db=redis_score_db,redis_mate_db=redis_mate_db)



mp = move_picker(redis_score_db=redis_score_db,redis_mate_db=redis_mate_db,
                 player='w',mate_depth = mate_depth)





def main():
    
    test_scholar_mate()
    #train_and_test_model()
    '''
    if not (persist_model):
        train_and_test_model()
    use_model()
    '''

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

    gam_an_obj = game_analyzer.game_analyzer(player=player,scores_file=scores_file)
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


    red_obj = populator(depth=score_depth,mate_depth=mate_depth,board=board,
                        redis_score_db=redis_score_db,redis_mate_db=redis_mate_db)
    red_obj.reset_and_fill_redis()
    #keys after this point: "['c4b5', 'h7h6']:r1bqkb1r/pppp1pp1/2n2n1p/1B2p2Q/4P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 5"
    cowsay.cow("making predictions")
    #mp.create_move_tree()
    
    nn.send_move_scores_to_redis(board)
    #keys after this point: "['c4e6', 'f6h5']"
    cowsay.cow("choosing move")
    #move = mp.highest_average_move(board=board)
    move = mp.find_forced_wins(board=board)
    print(move)
    print("Finish")

    return 0

def test_tree(board: chess.Board = chess.Board()):
    
    cowsay.cow("processing redis boards")


    red_obj = populator(depth=score_depth,mate_depth=mate_depth,board=board,
                        redis_score_db=redis_score_db,redis_mate_db=redis_mate_db)
    red_obj.reset_and_fill_redis()
    #keys after this point: "['c4b5', 'h7h6']:r1bqkb1r/pppp1pp1/2n2n1p/1B2p2Q/4P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 5"

    mp.create_move_tree()
    cowsay.cow("pruning tree")
    #move = mp.highest_average_move(board=board)
    move = mp.find_forced_wins(board=board)
    print(move)
    print("Finish")

    return 0


if __name__ == "__main__":
    main()