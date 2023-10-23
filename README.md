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

Features to add:
Add space advantage feature using coordinates of pawns and distance from respective sides, also check for chains based on attackers of same type and whether or not they are pawns

Todo:
Craft algorithim for move scoring
Create function in main for creating and training and model and one for using it
(Idea) Create auto factor analysis of features added based on values in some file
Still have to evaluate use case of when checkmate is within depth of moves or less

Notes:
Discovered my term for compression applied to trees for sequential games is known as minmax pruning I believe.
For scalability in the future, add a machine id to the redis keys for moves uploaded by specific instances
move_picker function compress has logical error and now eraeses db up to winning move, instead want all moves even if losing for analysis.
move_picker compress function needs abstraction, and line 269 is certainely a logical error as it deletes keys which are 2 moves long and therefore not fully compressed yet.


Future optomizations:
Use redis pipeline for mass key readings and uploads
Use opening book in begining