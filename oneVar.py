import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def plot_2d(results: np.array, expected:np.array, labels=[''], title='', block=False):

    plt.figure()

    if len(results):
        plt.plot(results.T[0], results.T[1], 'xr', label=labels[0])
    if len(expected):
        plt.plot(expected.T[0], expected.T[1], '-b', label=labels[1])

    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show(block=block)

class Regression:
    
    def __init__(self, alpha, epocas, file, labels = [], featuresNumber = 1):
        self.t = [np.random.rand() for _ in range(featuresNumber+1)]
        self.alpha = alpha
        self.epocas = epocas
        self.df = pd.read_csv(filepath_or_buffer=file, names=labels)
        self.labels = labels
        self.featuresNumber = featuresNumber
        self.E = [0 for _ in range(epocas)]

    def h(self, x = []):
        return self.t[0] + sum(self.t[i+1]*x[i] for i in range(self.featuresNumber))

    def plotData(self):
        plot_2d(results=np.array(self.df), expected=[], labels=self.labels, block=True)

    def plotRegression2d(self):
        plot_2d(results=np.array(self.df), expected=np.array([[self.df.T[:-1][i][0],self.h(self.df.T[:-1][i])] for i in range(len(self.df))]), labels=self.labels, block=True)

    def plotError(self):
        plot_2d(results=[], expected=np.array([[i,self.E[i]] for i in range(self.epocas)]), labels=['Época', 'Erro'], block=True)

    def execute(self):

        for epoca in range(0, self.epocas):

            Ex = [0 for j in range(self.featuresNumber+1)]

            for i in range(len(self.df)):
                error = self.h(self.df.T[:-1][i]) - self.df.T[i][-1]
                self.E[epoca] += np.power(error, 2) / (2 * len(self.df))
                Ex[0] += error / (len(self.df))
                for j in range(self.featuresNumber):
                    Ex[j+1] += (error * self.df.T[:-1][i][j]) / (len(self.df))

            for i in range(self.featuresNumber+1):
                self.t[i] = self.t[i] - self.alpha * Ex[i]

        print(self.t)

rg = Regression(0.01, 1000, "./data1.txt", ['population', 'profit'])
rg.execute()
rg.plotData()
rg.plotRegression2d()
rg.plotError()
