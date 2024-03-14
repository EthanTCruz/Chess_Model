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

https://stackoverflow.com/questions/69926198/generatordatasetopdataset-will-not-be-optimized-because-the-dataset-does-not-im