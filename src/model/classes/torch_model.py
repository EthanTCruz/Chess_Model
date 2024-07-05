import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Chess_Model.src.model.classes.MongoDBDataset import MongoDBDataset
from Chess_Model.src.model.config.config import Settings
from tqdm import tqdm
import torch.optim as optim

class FullModel(nn.Module):
    def __init__(self, input_planes, additional_features, output_classes=3):
        super(FullModel, self).__init__()




        self.conv1 = nn.Conv2d(in_channels=input_planes, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(256 * 8 * 8 + 64 * 8 * 8, 1024)  # Adjust input dimension to match concatenated features
        self.fc2 = nn.Linear(1024, output_classes)

        self.fc_additional = nn.Linear(additional_features, 64 * 8 * 8)
        
    def forward(self, bitboards, metadata):
        # print(f"After start shape: {bitboards.shape}")
        x = F.relu(self.conv1(bitboards))
        # print(f"After conv1: {x.shape}")

        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")

        x = F.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")

        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"After flatten: {x.shape}")

        # Process metadata through fc_additional
        metadata_processed = F.relu(self.fc_additional(metadata))
        # print(f"After fc_additional: {metadata_processed.shape}")

        # Flatten metadata_processed
        metadata_processed = metadata_processed.view(metadata_processed.size(0), -1)
        # print(f"After flatten metadata_processed: {metadata_processed.shape}")

        # Concatenate bitboards and metadata features
        combined_features = torch.cat((x, metadata_processed), dim=1)
        # print(f"After concatenation: {combined_features.shape}")

        x = F.relu(self.fc1(combined_features))
        # print(f"After fc1: {x.shape}")

        x = self.fc2(x)
        # print(f"After fc2: {x.shape}")

        x = F.log_softmax(x, dim=1)  # Softmax output
        # print(f"After softmax: {x.shape}")

        return x
    

class model_operator():
    def __init__(self,mdp):

        s = Settings()
        
        self.collectionName=s.training_collection_key
        self.mongoUrl=s.mongo_url
        self.dbName=s.db_name
        self.batch_size=s.nnBatchSize
        self.genBatchSize = s.nnGenBatchSize
        self.num_workers = s.num_workers

    def create_dataloader(self):
        dataset = MongoDBDataset(collectionName=self.collectionName, 
                        mongoUrl=self.mongoUrl, 
                        dbName=self.dbName, 
                        batch_size=self.genBatchSize)
        
        if len(dataset) == 0:
            print("Dataset is empty. Please check the data loading process.")
            return 0
        else:

            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            return dataloader
    
    def Create_and_Train_Model(self,
                               genBatchSize:int = 1024,
                           num_workers:int=0,
                          learning_rate:float = 0.001,
                          num_epochs:int = 5):
    
        dataloader = self.create_dataloader()
        shapes = self.calc_shapes()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # shapes[0][1] = number of bitboards, shapes[1][2] = number of metadata features
        model = FullModel(shapes[0][1], shapes[1][2]).to(device)

        criterion = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    
        for epoch in range(num_epochs):

            model.train()

            running_loss = 0.0
            total_samples = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_x1, batch_x2, batch_labels in progress_bar:

                # Move data to the appropriate device (CPU/GPU)
                batch_x1 = batch_x1.to(device)
                batch_x2 = batch_x2.to(device)
                batch_labels = batch_labels.to(device)
    
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_labels)
    
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                total_samples += batch_x1.size(0)  # Add the number of samples in the current batch
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, All samples processed.")
                  
            # Check if all samples were processed
            # if total_samples == len(dataset):
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, All samples processed.")
            # else:
            #     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Warning: Not all samples processed! Processed: {total_samples}, Expected: {len(dataset)}")


    def verify_dataloader_and_dataset(self,batch_size,num_workers: int = 0):


        dataset = MongoDBDataset(collectionName=self.collectionName, 
                mongoUrl=self.mongoUrl, 
                dbName=self.dbName, 
                batch_size=self.genBatchSize,
                mdp=self.mdp)
        # Ensure that the dataset is not empty before creating the DataLoader
        
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        else:
            print("Dataset is empty. Please check the data loading process.")

        # Debugging: Manually iterate through the dataset
        print("Manually iterating through the dataset:")
        for i in range(min(len(dataset), 5)):  # Check the first 5 items
            item = dataset[i]
            print(f"Item {i}: {item}")
        
        # Debugging: Iterate through the DataLoader
        print("Iterating through the DataLoader:")
        for batch in dataloader:
            print("Batch loaded")
            batch_x1, batch_x2, batch_labels = batch
            print("Batch shapes:", batch_x1.shape, batch_x2.shape, batch_labels.shape)
            break  # Break after the first batch for debugging
        
        # Debugging: Manually iterate through the dataset
        print("Manually iterating through the dataset:")
        for i in range(min(len(dataset), 5)):  # Check the first 5 items
            item = dataset[i]
            print(f"Item {i}: {item}")
        
        # Debugging: Iterate through the DataLoader
        print("Iterating through the DataLoader:")
        for batch in dataloader:
            print("Batch loaded")
            batch_x1, batch_x2, batch_labels = batch
            print("Batch shapes:", batch_x1.shape, batch_x2.shape, batch_labels.shape)
            break  # Break after the first batch for debugging


    def calc_shapes(self,  num_workers: int = 0):
        dataset = MongoDBDataset(collectionName=self.collectionName, 
                                mongoUrl=self.mongoUrl, 
                                dbName=self.dbName, 
                                batch_size=1,
                                mdp=self.mdp)
        
        # Ensure that the dataset is not empty before creating the DataLoader
        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        else:
            print("Dataset is empty. Please check the data loading process.")
            return
        
        # Debugging: Iterate through the DataLoader and return shapes of the first batch
        for batch in dataloader:
            batch_x1, batch_x2, batch_labels = batch

            return batch_x1.shape, batch_x2.shape, batch_labels.shape

