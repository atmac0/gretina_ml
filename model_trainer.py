import tensorflow as tf
from tensorflow import keras

from data_wrangler import DataWrangler

def train_model(data):

    train_inputs, train_labels, test_inputs, test_labels = data.get_training_dataset()
    print("Successfully obtained training input and labels")    

    n_hidden_layer = 16 # size of hidden layer
    n_output_layer = 2

    max_interactions = data.get_max_interactions()
    dimensionality_of_interaction = data.get_dimensionality_of_interaction()
    
    model = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.LSTM(n_output_layer, return_sequences=True), input_shape=(max_interactions, dimensionality_of_interaction)),
        keras.layers.Dense(32, activation='tanh'),
        keras.layers.Bidirectional( keras.layers.LSTM(n_hidden_layer, activation='tanh') ),
        keras.layers.Dense(n_output_layer, activation='softmax')
    ])    

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])    

    print("Model go beep boop")
    model.fit(train_inputs, train_labels, epochs=100)

    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    return model

def get_model(save=False):

    files = ['out_1173.csv', 'out_1332.csv', 'out_2505.csv']
    max_clusters_per_file = 6000

    data = DataWrangler(files, max_clusters_per_file=max_clusters_per_file)

    model = train_model(data)

    if(save):
        model.save('model')
    
    return model
