from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import  OAuth2PasswordRequestForm

from fastapi.middleware.cors import CORSMiddleware

import json
import chess


from Chess_Model.src.model.classes.potential_board_populator import populator
from  Chess_Model.src.model.classes.game_analyzer import game_analyzer
import chess
from Chess_Model.src.model.classes.pgn_processor import pgn_processor
from Chess_Model.src.model.classes.nn_model import neural_net
import os
import cowsay
from Chess_Model.src.model.classes.move_picker import move_picker
import redis
import hiredis
from Chess_Model.src.model.config.config import Settings
import uvicorn


s = Settings()
ModelFilePath=s.ModelFilePath
ModelFilename=s.ModelFilename
scores_file = s.scores_file
pgn_file = s.pgn_file
games_csv_file = s.games_csv_file
predictions_board = s.predictions_board

epochs = s.nnEpochs
batch_size = s.nnBatchSize
test_size = s.nnTestSize

persist_model = s.persist_model
score_depth = s.score_depth

#pgn_file=s.samplePgn


nn = neural_net(filename=scores_file,target_feature='w/b',
                test_size=test_size,ModelFilename = ModelFilename,
                ModelFilePath=ModelFilePath,player='w',
                predictions_board=predictions_board,epochs=epochs,
                trainModel=s.trainModel,batch_size=batch_size)



mp = move_picker(neuralNet=nn)




# FastAPI instance
app = FastAPI()

# OAuth2



@app.post("/aimove")
async def login(fen:str):
    board = chess.Board(fen=fen)
    move = mp.use_model(board=board)
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
