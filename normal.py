# Tarefa - Regressão Linear com Uma e Múltiplas Variáveis
# Parte 3 - Equação Normal
#
## Arthur Lopes Sabioni
## Julia Bindi Alencar de Jesus

# Tarefa - Regressão Linear com Uma e Múltiplas Variáveis
# Parte 2 - Regressão linear com múltiplas variáveis
#
## Arthur Lopes Sabioni
## Julia Bindi Alencar de Jesus

import pandas as pd
import numpy as np

class NormalRegression:
    
    def __init__(self, file, labels = []):
        self.t = [np.random.rand() for _ in range(len(labels))]
        self.df = pd.read_csv(filepath_or_buffer=file, names=labels)
        self.labels = labels

    def execute(self):
        X = [[1, x[0], x[1]] for x in np.array(self.df[self.labels[0:2]])]
        Y = np.array(self.df[self.labels[2]])
        self.t = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)), Y)

if __name__ == "__main__":
    nr = NormalRegression("./data2.txt", ['size', 'rooms', 'value'])
    nr.execute()
    print(nr.t)
