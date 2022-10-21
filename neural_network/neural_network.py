import keras_tuner as kt
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from pandas import read_csv
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, model_from_json

# Params

# k-fold
k = 5  
# seed for cross validation
seed = 0 

test_size = 0.33  

epochs = 10

validation_split = 0.2

def load_data(data_path):
	# Load and prepare data set
	dataframe = read_csv(data_path)
	dataset = dataframe.values

	X = dataset[:,0:11].astype(float)
	Y = dataset[:,-1]

	print("Data succesfully loaded.")

	return X,Y

data_path_train = "../data/project_train.csv"
data_path_test = "../data/project_test.csv"

X, Y = load_data(data_path_train)
X_test, _ = load_data(data_path_test)

def evaluate_cv(model):
	# Evaluate a model using k-fold cross validation (to compare models internally)
    kfold = KFold(n_splits= k, shuffle=True, random_state= seed)
    cvscores = []

    for train, test in kfold.split(X, Y):
        # split into train and test data 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state = seed)

        model.fit(X[train], Y[train], epochs=100, batch_size=2, verbose=0)
        
        # evaluate the model
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# ---------------------------- Hypertuning ----------------------------
def model_builder(hp):

	# Define the model: model architecture and hyperparameter space.

	model = Sequential()
	# Input layer ------------- 
	# Tune the number of units
	hp_first_unit = hp.Int('first_unit', min_value = 8, max_value = 60, step = 4)
	model.add(Dense(units = hp_first_unit, input_shape = (X.shape[1],), activation = 'relu'))
	# Tune the dropout rate 
	hp_dropout_0 = hp.Float('rate_1', min_value = 0.0, max_value = 0.5, step = 0.1)
	model.add(Dropout(hp_dropout_0))


	# Layer 1 ----------------- 
	# Tune the number of units 
	hp_units_1 = hp.Int('units_1', min_value = 8, max_value = 60, step = 4)
	# Tune the activation function
	hp_activation_1=hp.Choice('dense_activation_1',
		values=['relu', 'sigmoid'])
	model.add(Dense(units=hp_units_1, activation= hp_activation_1))
	# Tune the dropout rate 
	hp_dropout_1 = hp.Float('rate_1', min_value = 0.0, max_value = 0.5, step = 0.1)
	model.add(Dropout(hp_dropout_1))

	# Layer 2 ----------------- 
	# Tune the number of units in the each dense layer
	hp_units_2 = hp.Int('units_2', min_value = 8, max_value = 60, step = 4)
	# Tune the activation function
	hp_activation_2=hp.Choice('dense_activation_2',
		values=['relu', 'sigmoid'])
	model.add(Dense(units=hp_units_2, activation= hp_activation_2))
	# Tune the dropout rate in the each dense layer
	hp_dropout_2 = hp.Float('rate_2', min_value = 0.0, max_value = 0.5, step = 0.1)
	model.add(Dropout(hp_dropout_2))


	# Add dense output layer
	model.add(Dense(1,  activation = 'sigmoid'))

	# Tune the learning rate for the optimizer
	# Choose an optimal value from 0.01, 0.001, or 0.0001
	hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

	model.compile(optimizer = Adam(learning_rate = hp_learning_rate),
				loss = 'binary_crossentropy',   
				metrics = ['accuracy'],
				)
	return model

tuner = kt.BayesianOptimization(model_builder,
                     objective='val_accuracy',
                     directory='neural_network_finetuning',
                     project_name='SF2935_binary_classification')

# Optimize the hyperparameter search for hypermodel
tuner.search(X, Y, epochs = epochs, validation_split = validation_split)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"Hypertuned parameters: {best_hps.values}")

hypertuned_model = tuner.hypermodel.build(best_hps)
evaluate_cv(hypertuned_model)


# # serialize model to JSON
# model_json = hypertuned_model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# hypertuned_model.save_weights("model.h5")
# print("Saved model to disk")



# ---------------------------- Prediction ---------------------------- 
# Load tuned model and start final training to predict

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


Y_prediction = loaded_model.predict(X_test)

Y_prediction = (Y_prediction > 0.5).astype(int)

pd.DataFrame(Y_prediction).to_csv('../predictions/predictions_NN.csv', index = False,  header=None)
