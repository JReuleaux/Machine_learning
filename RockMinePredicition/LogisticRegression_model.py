from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from Rock_mine_data_process import load_and_process_data
import numpy as np

data_path  = "C:/Users/juria_rfq14t7/Documents/RockMinePredicition/Data/Copy of sonar data.csv"
rock_mine_predictor = load_and_process_data(data_path)
x_train, x_test, y_train, y_test = rock_mine_predictor.training_set_up()

class model:

    def __init__(self):
        self.set_up = self.model_set_up()    
    
    #Model training using Logistic regression
    def model_set_up(self):
        model = LogisticRegression()
        return model
    
    #Training the model using the training data
    def train_model(self):
        self.set_up.fit(x_train, y_train)

    def check_accuracy(self):
        #Evaluating the model by checking its accuracy on the training data
        x_train_prediction = self.set_up.predict(x_train)
        training_data_accuracy = accuracy_score(x_train_prediction, y_train)
        print(f"Data training accuracy: {training_data_accuracy:.3f}")


        #Evaluating the model by checking its accuracy on the test data
        x_test_prediction = self.set_up.predict(x_test)
        test_data_accuracy = accuracy_score(x_test_prediction, y_test)
        print(f"Data test accuracy: {test_data_accuracy:.3f}")
    
    def predict(self, value):
        numpy_data = np.asarray(value)
        reshaped = numpy_data.reshape(1, -1)
        prediction = self.set_up.predict(reshaped)
        if (prediction[0] == "R"):
            print("its a rock")
        else:
            print("ITS A MINE!")