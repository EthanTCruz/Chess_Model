import redis_populator
import game_analyzer
import chess

def main():
    board = chess.Board()
    red_obj = redis_populator.populator(depth=3,board=board)
    red_obj.reset_and_fill_redis()
    gam_an_obj = game_analyzer.game_analyzer(player='w')
    gam_an_obj.process_redis_boards()

def test_model():
    return 0
def train_model():
    return 0

if __name__ == "__main__":
    main()