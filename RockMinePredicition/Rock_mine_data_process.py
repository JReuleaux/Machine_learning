import pandas as pd
from sklearn.model_selection import train_test_split

class load_and_process_data:

    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        #Loading data set (.csv) into a pandas dataframe.
        sonar_data = pd.read_csv(data_path, header=None)
        print(sonar_data.shape)
        print(sonar_data.describe()) #Describing some statistical measures
        print(sonar_data[60].value_counts()) #Checking how many rocks(R) and how many mines(M) there are.
        print(sonar_data.groupby(60).mean())
        return sonar_data
    
    def training_set_up(self):
        #Seperating the data and the labels
        x = self.data.drop(columns=60, axis=1)
        y = self.data[60]
        #setting up the train and test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)
        return x_train, x_test, y_train, y_test

