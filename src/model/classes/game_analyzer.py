import chess
import redis
import hiredis
import csv
import os
import ast
from src.model.classes.scorer import boardEval
from tqdm import tqdm
from src.model.config.config import Settings

Start_value =  "['d2d4', 'e7e6', 'c1h6']:rnbqkbnr/pppp1ppp/4p2B/8/3P4/8/PPP1PPPP/RN1QKBNR b KQkq - 1 2"

class game_analyzer:
    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        if "output_file" not in kwargs:
            self.output_file="./data/data.csv"
        else:
            self.output_file = kwargs["output_file"]
        

        if "persist_data" not in kwargs:
            self.persist_data = False
        else:
            self.persist_data = kwargs["persist_data"]

        if "player" not in kwargs:
            self.player = "NA"
        else:
            self.player = kwargs["player"]
        
        if "redis_score_db" not in kwargs:
            self.r_score = redis.Redis(host='localhost', port=6379,db=1)
        else:
            self.r_score = redis.Redis(host=s.redis_host, port=s.redis_port,db=int(s.redis_score_db))




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
        with open(csv_file) as f:
            total_lines = sum(1 for line in f)
        with open(self.output_file, 'a', newline='') as gameEvalfile:
            writer = csv.writer(gameEvalfile)
            # find total number of lines in the file


            with open(csv_file, 'r') as fenfile:
                csv_reader = csv.reader(fenfile)
                for row in tqdm(csv_reader, total=total_lines):
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
                            pass
                        except AttributeError:
                            print(row) 
                            raise Exception("here")



    def process_redis_boards(self):
        self.create_csv()
        cursor = 0
        count = 1000
        with open(self.output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            while True:
                # Scan the Redis database using the cursor
                cursor, keys = self.r_score.scan(cursor=cursor, count=count)
                # Print the retrieved keys
                for key in keys:
                    values = key.decode('utf-8').split(":")
                    moves = [values[0]]
                    fen = values[1]
                    scores = self.evaluate_board(fen=fen).values()
                    if self.r_score.get(key) == 'b':
                         scores["w/b"] = 'b'
                    elif self.r_score.get(key) == 'w':
                        scores["w/b"] = 'w'     
                    row = moves + list(scores)
                    writer.writerow(row)
                    self.r_score.delete(key)


                # Break the loop if the cursor returns to the start
                if cursor == 0:
                     break




    def evaluate_board(self,fen,victor="NA"):
        evaluator = boardEval(fen=fen,player=self.player)
        return evaluator.get_board_scores(victor=victor)

    def create_csv(self):
        # Check if the file exists and remove it
        if os.path.exists(self.output_file) and not self.persist_data:
            os.remove(self.output_file)

        # Create a new CSV file with the column headers
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            scorer_obj = boardEval()
            features = scorer_obj.get_features()
            writer.writerow(features)

