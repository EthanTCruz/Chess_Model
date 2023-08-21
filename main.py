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

ModelFilePath="./"
ModelFilename="chess_model"
scores_file = "./data/data.csv"
pgn_file = "./pgn/Adams.pgn"
games_csv_file = "./data/games.csv"
predictions_board = './data/predictions.csv'
redis_score_db = redis.Redis(host='localhost', port=6379,db=1)
redis_mate_db = redis.Redis(host='localhost', port=6379,db=2)
persist_model = True

nn = neural_net(filename=scores_file,target_feature='w/b',
                test_size=0.3,ModelFilename = ModelFilename,
                ModelFilePath=ModelFilePath,player='w',
                predictions_board=predictions_board,epochs=100,
                redis_score_db=redis_score_db,redis_mate_db=redis_mate_db)



mp = move_picker(redis_score_db=redis_score_db)





def main():
    
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")
    board.push_san("Nc6")
    board.push_san("Qh5")
    board.push_san("Nf6")
    use_model(board=board)
    #train_and_test_model()
    '''
    if not (persist_model):
        train_and_test_model()
    use_model()
    '''





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


    red_obj = populator(depth=2,board=board,
                        redis_score_db=redis_score_db,redis_mate_db=redis_mate_db)
    red_obj.reset_and_fill_redis()
    #keys after this point: "['c4b5', 'h7h6']:r1bqkb1r/pppp1pp1/2n2n1p/1B2p2Q/4P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 5"
    cowsay.cow("making predictions")

    
    nn.send_move_scores_to_redis(board)
    #keys after this point: "['c4e6', 'f6h5']"
    cowsay.cow("choosing move")
    move = mp.highest_average_move(board=board)
    print(move)
    print("Finish")

    return 0

if __name__ == "__main__":
    main()