# Clustering Model
# Instantiated with pretrained linear SVM with weights corresponding to learned co-efficient weights from the loss model

import pickle
import keras
import tensorflow as tf
model_reps = ['title', 'body', 'titlebody','title_lemmas', 'body_lemmas',
                   'titlebody_lemmas','title_entities','body_entities',
                    'titlebody_entities','bert','bert_ent']

#model_reps = ['title', 'body', 'titlebody','title_lemmas', 'body_lemmas',
#                    'titlebody_lemmas','title_entities','body_entities',
#                    'titlebody_entities','bert']

time_reps = ["NEWEST_TS", "OLDEST_TS", "RELEVANCE_TS"]

class Model:
    def __init__(self, path):
        self.weights = {}
        self.bias = 0
        self.path = path

    def load_NN(self):
        modelNN = self.create_NNmodel()
        modelNN.load_weights(self.path)
        return (modelNN)

    def create_NNmodel(self):
        input_val =15
        m =  tf.keras.models.Sequential([
            
            
            keras.layers.Dense(25, activation='relu', input_shape=(input_val,)),
            keras.layers.Dense(input_val, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(5, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),])
        
        m.build((500, input_val))
    
        m.compile(optimizer='adam',
                loss=tf.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.metrics.BinaryAccuracy()])
    

        return m

    def load_SVMweights(self, modelkeys):
        linear_lossmodel = pickle.load(open(self.path, 'rb'))

        July30vlaues = [0.04198982226306214, -0.9084837441514324, -0.8181088282025044, 0.5482390666248662, -0.24638171323402602,0.3146409567515711,
        0.01867149494390219, -0.1177425777405432, 0.0, 0.1897917519286132, -0.004771956028903901, -2.0773264325500946, 0.11020307187183676, 0.0]
        
        repovalues = [-1.9079906, 1.0086491 , 3.8883567, 0.32604426, -1.7037395, 0.74489218, 1.4310931, -0.73693264, 2.4383616, 2.4, 1, 2.1834517, 5.0244422, 0]
        lossvalues = linear_lossmodel.coef_[0]

        losskeys = modelkeys
 
        self.weights = dict(zip(losskeys, repovalues))
        return (self.weights)


