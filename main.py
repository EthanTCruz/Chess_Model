import redis_populator
import game_analyzer
import chess
from pgn_processor import pgn_processor

def main():
    print("start")
    gam_an_obj = game_analyzer.game_analyzer(player='w',output_file="data.csv")
    pgn_obj = pgn_processor(pgn_file='sample.pgn',csv_file='games.csv')
    pgn_obj.pgn_fen_to_csv()
    gam_an_obj.process_csv_boards(csv_file="games.csv")


    
    '''
    board = chess.Board()
    red_obj = redis_populator.populator(depth=2,board=board)
    red_obj.reset_and_fill_redis()
    gam_an_obj.process_redis_boards()
    
    
    '''
    print("finish")



def test_model():
    return 0
def train_model():
    return 0

if __name__ == "__main__":
    main()