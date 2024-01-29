"# chess_model" 
```docker run --name my-redis -p 6379:6379 -d redis```
```pip install -r requirments.txt```
```python main.py```
Adams.pgn:
5/4/23
Test loss: 0.47839900851249695
Test accuracy: 0.7689734101295471

5/5/23
Test loss: 0.4322868287563324
Test accuracy: 0.8030847311019897

5/28/23
Test loss: 0.4277477264404297
Test accuracy: 0.811082124710083

01/07/24 - large tweeks and set seed
Test loss: 0.5457004308700562
Test accuracy: 0.8052128553390503

01/08/24
Test loss: 0.37744206190109253
Test accuracy: 0.8561661839485168

01/10/24
Test loss: 0.4207151234149933
Test accuracy: 0.8471933007240295

01/19/24 - implemented batch processing to save memory
Test loss: 0.44103720784187317
Test accuracy: 0.8269000053405762

Features to add:
Add space advantage feature using coordinates of pawns and distance from respective sides, also check for chains based on attackers of same type and whether or not they are pawns

Todo:
Craft algorithim for move scoring
Create function in main for creating and training and model and one for using it
(Idea) Create auto factor analysis of features added based on values in some file
Still have to evaluate use case of when checkmate is within depth of moves or less
Get rid of Mate redis db

Notes:
Discovered my term for compression applied to trees for sequential games is known as minmax pruning I believe.
For scalability in the future, add a machine id to the redis keys for moves uploaded by specific instances
move_picker function compress has logical error and now eraeses db up to winning move, instead want all moves even if losing for analysis.
move_picker compress function needs abstraction, and line 269 is certainely a logical error as it deletes keys which are 2 moves long and therefore not fully compressed yet.


Future optomizations:
Use redis pipeline for mass key readings and uploads
Use opening book in begining

docker tag chess_model:v1 ethancruz/chess_model
docker push ethancruz/chess_model:latest


W
Batch size: 50, epochs: 500
Test loss: 0.3644968867301941
Test accuracy: 0.8490626215934753

Batch size: 50, epochs: 100
Test loss: 0.3637749254703522
Test accuracy: 0.8343748450279236

Batch size: 50, epochs: 250
Test loss: 0.36239126324653625
Test accuracy: 0.8449500799179077

Batch size: 50, epochs: 300
Test loss: 0.3614208698272705
Test accuracy: 0.8468728065490723

pin(color: chess.Color, square: chess.Square)→ SquareSet[source]
Detects an absolute pin (and its direction) of the given square to the king of the given color.

import chess

board = chess.Board("rnb1k2r/ppp2ppp/5n2/3q4/1b1P4/2N5/PP3PPP/R1BQKBNR w KQkq - 3 7")
board.is_pinned(chess.WHITE, chess.C3)
True
direction = board.pin(chess.WHITE, chess.C3)
direction
SquareSet(0x0000_0001_0204_0810)
print(direction)
. . . . . . . .
. . . . . . . .
. . . . . . . .
1 . . . . . . .
. 1 . . . . . .
. . 1 . . . . .
. . . 1 . . . .
. . . . 1 . . .
Returns a set of squares that mask the rank, file or diagonal of the pin. If there is no pin, then a mask of the entire board is returned.

is_pinned(color: chess.Color, square: chess.Square)→ bool[source]
Detects if the given square is pinned to the king of the given color.