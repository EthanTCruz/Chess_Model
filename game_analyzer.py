import chess
import redis
import hiredis
import csv
import os
import ast
import scorer

Start_value =  "['d2d4', 'e7e6', 'c1h6']:rnbqkbnr/pppp1ppp/4p2B/8/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2"

class game_analyzer:
    def __init__(self,output_file,player="NA",persist_data=False) -> None:
        self.output_file = output_file
        self.persist_data = persist_data
        self.player = player


    def process_single_board(self,board_key,victor="NA"):
        with open(self.output_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    values = board_key.split(":")
                    moves = [values[0]]
                    fen = values[1]
                    scores = list(self.evaluate_board(fen=fen,victor=victor).values())
                    row = moves + scores
                    writer.writerow(row)
    
    def process_csv_boards(self,csv_file):
        self.create_csv()
        with open(self.output_file, 'a', newline='') as gameEvalfile:
            writer = csv.writer(gameEvalfile)
            with open(csv_file, 'r') as fenfile:
                csv_reader = csv.reader(fenfile)
                for row in csv_reader:
                        if row == "":
                             return 1
                        try:
                            victor = row[2]
                            fen = row[1]
                            moves = [row[0]]
                            scores = list(self.evaluate_board(fen=fen,victor=victor).values())

                            #row = [moves,scores]
                            moves += scores
                            writer.writerow(moves)
                        except AttributeError:
                            print(row) 
                            raise Exception("here")


    def process_redis_boards(self):
        if os.path.exists(self.output_file):
            first_line = ''
            with open(self.output_file, 'r') as csv_file:
                reader = csv.reader(csv_file)
                first_line = next(reader)

            os.remove(self.output_file)

            with open(self.output_file, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(first_line)

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
                    scores = self.evaluate_board(fen=fen).values()
                    if r.get(key) == 'b':
                         scores["w/b"] = 'b'
                    elif r.get(key) == 'w':
                        scores["w/b"] = 'w'     
                    row = moves + list(scores)
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
            scorer_obj = scorer.boardEval()
            features = scorer_obj.get_features()
            writer.writerow(features)

