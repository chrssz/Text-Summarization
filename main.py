import tensorflow as tf
import keras
from collections import defaultdict
from keras._tf_keras.keras.layers import TextVectorization #for tokenization
from keras._tf_keras.keras.utils import pad_sequences
from datasets import load_dataset


def loadData() -> None:
    #load the 'gem' dataset
    data = load_dataset("gem", "common_gen")
    #x-input, y-target
    X_train,y_train = data["train"]["concepts"], data["train"]["target"]
    x_test,y_test = data["test"]["concepts"], data["test"]["target"]
    validation = data["validation"]
    
    X_train = preProcess(X_train)
    x_test = preProcess(x_test)
    return X_train,y_train,x_test,y_test,validation

def preProcess(l)->list:
    #preprocess text
    token = defaultdict(int)

    def tokenize(l)->list: #helper function to tokenize text
        tokenNo = 0
        if isinstance(l,list):
            for sublist in l:
                for string in sublist:
                    if string in token:
                        string = token[string]
                    else:
                        token[string] += tokenNo
                        str = token[string]
                        tokenNo+=1
            return l
        
        elif isinstance(l,str): #target is a str
            lower = l.lower()
            new_list = []
            for char in lower:
                if char == ' ':
                    continue
                if char in token:
                    new_list.append(token[char])
                else:
                    token[char] += tokenNo
                    new_list.append(token[char])
                    tokenNo +=1
            return new_list
        
    return tokenize(l)

def NeuralNetwork():
    #calling this function inits the neural network
    #Using a Convolutional 1D layer
    model = keras.Sequential()
    # Define hyperparameters for Embedding layer
    MAX_FEATURES = 10000  # Experiment with this value
    EMBEDDING_DIM = 128  # Experiment with this value
    #model.add(keras.layers.TextVectorization(max_tokens=1000,output_mode=int))
    model.add(keras.layers.Embedding(input_dim=MAX_FEATURES, output_dim=EMBEDDING_DIM))

    model.add(keras.layers.Conv1D(filters=30,kernel_size=5,activation='relu'))
    model.add(keras.layers.MaxPool1D(pool_size=2,strides=2))
    model.add(keras.layers.Conv1D(filters=15,kernel_size=3,activation='relu'))
    model.add(keras.layers.MaxPool1D(pool_size=2,strides=2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(units=200,activation='relu'))
    model.add(keras.layers.Dense(units=100,activation='relu'))
    model.add(keras.layers.Dense(units=50,activation='softmax'))
    
    model.compile(optimizer='adam',loss = 'categorical_crossentropy')

    return model


X_train,y_train, x_test, y_test, validation = loadData()

model = NeuralNetwork()

model.fit(X_train,y_train, epochs=10,batch_size=150,validation_data=validation)
 
loss,acc = model.evaluate(x_test,X_train)

print(f'Test loss: {loss}')
print(f'Test accuracy: {acc}')