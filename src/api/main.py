from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import  OAuth2PasswordRequestForm

from fastapi.middleware.cors import CORSMiddleware

import json
import chess

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
import uvicorn


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





# FastAPI instance
app = FastAPI()

# OAuth2

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
    return move

@app.post("/aimove")
async def login(fen:str):
    board = chess.Board(fen=fen)
    move = use_model(board=board)[0]
    print(move)
    return {"move" : move}




# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
