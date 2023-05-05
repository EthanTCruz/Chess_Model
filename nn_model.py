import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class neural_net():

    def __init__(self,filename,target_feature = "w/b",test_size=.2,random_state=42) -> None:
        self.target_feature = target_feature
        self.data = pd.read_csv(filename)
        self.test_size=test_size
        self.random_state = random_state

        X,Y = self.clean_data()
        self.partition_and_scale(X=X,Y=Y)





    def clean_data(self):
        # Remove the moves column, as it's not a useful feature for the neural network
        self.data = self.data.drop(columns=['moves(id)'])

        # Encode the target variable (w/b) as 0 or 1
        self.data[self.target_feature] = self.data[self.target_feature].apply(lambda x: 1 if x == 'w' else 0)
        # One-hot encode the 'game time' feature

        # Split the data into features and target
        X = self.data.drop(columns=[self.target_feature])
        Y = self.data[self.target_feature]

        return X,Y



    def partition_and_scale(self,X,Y):
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)

        # Scale the feature data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    
    def create_and_evaluate_model(self):
        # Create a simple neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(self.X_train, self.Y_train, batch_size=32, epochs=100, validation_split=0.2)
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(self.X_test, self.Y_test)

        print("Test loss:", loss)
        print("Test accuracy:", accuracy)
        return model
    

#implement following code later to save models
'''
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.
'''