"# chess_model" 
```docker run --name my-redis -p 6379:6379 -d redis```
```pip install -r requirments.txt```
```python main.py```
Adams.pgn:
5/4/23
Test loss: 0.47839900851249695
Test accuracy: 0.7689734101295471

Features to add:
binary: has bishop pair (opp,player)

Todo:
Switch from using redis to using csv
Craft tree and prune losing moves from it
Score leaves
Craft algorithim for tree traversal to select moves based on opponent choosing best move for themself