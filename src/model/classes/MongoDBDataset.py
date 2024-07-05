import torch
from torch.utils.data import Dataset, DataLoader
from Chess_Model.src.model.classes.mongo_functions import mongo_data_pipe
from pymongo import MongoClient

class MongoDBDataset(Dataset):
    def __init__(self, mdp,collectionName, mongoUrl, dbName, batch_size=1):

        self.batch_size = batch_size
        

        client = MongoClient(mongoUrl)
        db = client[dbName]
        self.collection = db[collectionName]

        
        self.mdp = mdp

        
        self.data = []

        # Use a process-safe way to load data
        self._load_data()
        
        # Fetch data with a progress bar

    def _load_data(self):
        try:
            for doc in self.mdp.iteratingFunctionScaled(collection=self.collection,
                                               batch_size=self.batch_size):
                self.data.append(doc)
            print(f"Loaded {len(self.data)} documents.")
        except Exception as e:
            print("Error during data loading:", e)
            

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            if idx >= len(self.data):
                raise IndexError("Index out of range")
            
            doc = self.data[idx]
            positions_data = doc['positions_data']
            metadata = doc['metadata']
            game_results = doc['game_results']
            
            # print(f"Returning item {idx} from dataset")
            
            return (torch.tensor(positions_data, dtype=torch.float32),
                    torch.tensor(metadata, dtype=torch.float32),
                    torch.tensor(game_results, dtype=torch.float32))
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            raise

        
