import pandas as pd
import numpy as np

from numpy.linalg import inv

class Calculator:
    def __init__(self, X, y, np_y, np_y_n):
        self.entropies = {}
        self.y = y
        self.X = X
        self.np_y = np_y
        self.np_y_n = np_y_n
        self.X1 = self.X.copy()
        self.X1.columns = [i+1 for i in range(len(self.X1.columns))]
        self.X1[0] = 1
        self.X1 = self.X1.reindex(list(self.X1.columns.sort_values()), axis=1)
        for column in self.X.columns:
            self.entropies[column] = self.calculate_entropy(self.X[column])

    def calculate_entropy(self, column):
        value_counts = column.value_counts()
        non_zero_counts = value_counts[value_counts != 0]
        ent = -(non_zero_counts / len(column) * np.log2(non_zero_counts / len(column))).sum()
        return ent

    def calculate_joint_entropy(self, column_1, column_2):
        joint_probs = column_1.groupby([column_1, column_2]).size() / len(column_1)
        joint_entropy = -(joint_probs * np.log2(joint_probs)).sum()
        return joint_entropy

    def calculate_mutual_information(self, entropy_1, entropy_2, column_1, column_2):
        joint_entropy = self.calculate_joint_entropy(column_1, column_2)
        mutual_information = entropy_1 + entropy_2 - joint_entropy
        return mutual_information

    def calculate_total_mi(self, target):
        target = pd.DataFrame(target, columns=['y'])['y']
        self.entropies['target'] = self.calculate_entropy(target)
        
        mutual_information_data = pd.DataFrame(columns=['column', 'value'])
        mutual_information_data['column'] = self.X.columns
        mutual_information_data['value'] = mutual_information_data['column'].apply(lambda x: self.calculate_mutual_information(self.entropies['target'], self.entropies[x], target, self.X[x]))

        # mutual_information_data_pd = pd.DataFrame(mutual_information_data)
        return mutual_information_data['value'].sum()


    def calculate_R2(self, df_Y):
        XT = self.X1.T.reset_index(drop=True)
        
        b = inv(XT @ self.X1) @ XT @ df_Y
        bT = b.T.reset_index(drop=True)

        temp1 = bT @ XT @ df_Y
        SS = (temp1 - (1/len(self.X1))*((self.X1[0].T@df_Y)**2))
        SS_error = (df_Y.T @ df_Y) - (temp1)
        R2 = SS/ (SS + SS_error)

        return SS[0][0], SS_error[0][0], R2[0][0]
    
    def calculate_R2_score_apply(self, individual):
        # Calculate the R2 Score of the individual
        _, _, r2_score = self.calculate_R2(pd.DataFrame(individual))

        return r2_score


    def calculate_accuracy(self, individual):
        # Create a for loop to calculate accuracy
        accuracy = 0
        for i in range(0, self.np_y_n):
            if self.np_y[i] == individual[i]:
                accuracy = accuracy + 1

        # Return accuracy
        return accuracy/self.np_y_n


def convert_str_to_list(data):
    # Remove brackets and split by spaces
    data = data.replace('[', '').replace(']', '')
    data_list = data.split()

    # Convert string to int
    data_list = [int(i) for i in data_list]

    return data_list

def convert_list_to_str(data):
    # Convert list to string
    data = str(data)

    # Remove brackets and split by spaces
    data = data.replace(', ', ' ')

    return data