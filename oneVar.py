import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def plot_2d(results: np.array, expected = [], labels=[''], title='', block=False):

    plt.figure()

    plt.plot(results.T[0], results.T[1], 'xr', label=labels[0])
    if len(expected):
        plt.plot(expected.T[0], expected.T[1], '-b', label=labels[1])

    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show(block=block)

class Regression:
    
    def __init__(self, alpha, epocas, file, labels = [], featuresNumber = 1):
        self.t = [np.random.rand() for j in range(featuresNumber+1)]
        self.alpha = alpha
        self.epocas = epocas
        self.df = pd.read_csv(filepath_or_buffer=file, names=labels)
        self.featuresNumber = featuresNumber

    def h(self, x = []):
        return self.t[0] + sum(self.t[i]*x[i] for i in range(self.featuresNumber))

    def execute(self):

        E = [0 for i in range(self.epocas)]

        for epoca in range(0, self.epocas):

            Ex = [0 for j in range(self.featuresNumber+1)]

            for i in range(len(self.df.T[0])):
                error = self.h(self.df.T[:-1][i])
                E[epoca] += (error ** 2) / (2 * len(self.df))
                Ex[0] += error
                for j in range(self.featuresNumber):
                    Ex[j+1] += (error * self.df.T[:-1][i][j]) / (len(self.df))

            for i in range(self.featuresNumber+1):
                self.t[i] = self.t[i] + self.alpha * Ex[i]

        plot_2d(results=np.array(self.df), expected=[], labels=['população', 'lucro'], block=True)

rg = Regression(0.1, 100, "./data2.txt", ['population', 'profit','dd'])
rg.execute()

