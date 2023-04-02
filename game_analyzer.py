import chess
import redis
import hiredis
import csv
import os
import ast
import scorer

Start_value =  "['d2d4', 'e7e6', 'c1h6']:rnbqkbnr/pppp1ppp/4p2B/8/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2"

class game_analyzer:
    def __init__(self,player) -> None:
        self.features = ["moves(id)","pawns","knights","bishops","rooks","queens","my moves","opp moves","checkmate","game time","w/l"]
        self.output_file = "data.csv"
        self.persist_data = False
        self.player = player
        self.create_csv()

    def process_single_board(self,board_key):
        with open(self.output_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    values = board_key.split(":")
                    moves = [values[0]]
                    fen = values[1]
                    scores = self.evaluate_board(fen=fen)
                    row = moves + scores
                    writer.writerow(row)
    
    def process_redis_boards(self):
        r = redis.Redis(host='localhost', port=6379)
        cursor = 0
        count = 10
        with open(self.output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            while True:
                # Scan the Redis database using the cursor
                cursor, keys = r.scan(cursor=cursor, count=count)
                # Print the retrieved keys
                for key in keys:
                    values = key.decode('utf-8').split(":")
                    moves = [values[0]]
                    fen = values[1]
                    scores = self.evaluate_board(fen=fen)
                    row = moves + scores
                    writer.writerow(row)
                    r.delete(key)


                # Break the loop if the cursor returns to the start
                if cursor == 0:
                    break

    def evaluate_board(self,fen,victor="NA"):
        evaluator = scorer.boardEval(fen=fen,player=self.player)
        return evaluator.get_board_scores(victor=victor)

    def create_csv(self):
        # Check if the file exists and remove it
        if os.path.exists(self.output_file) and not self.persist_data:
            os.remove(self.output_file)

        # Create a new CSV file with the column headers
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.features)

test = game_analyzer(player='w')
test.process_single_board(board_key=Start_value)