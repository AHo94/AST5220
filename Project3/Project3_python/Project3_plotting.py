import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# Change directory based based on work place, i.e home or at UiO computer.
file_directory = 'C:/Users/Alex/Documents/AST5220/AST5220_Projects/Project3/Project3_cpp/build-Project3-Desktop_Qt_5_7_0_MinGW_32bit-Debug'

class Plotter():
	def __init__(self, savefile):
		self.savefile = savefile	# If True, saves the plots to a file

	def read_data(self, filename_open):
		""" Reads data and saves values to variables and arrays"""
		filename = open(os.path.join(file_directory, filename_open), 'r')
		i = 0
		data = []
		for line in filename:
			data_set = line.split()
			if i != 0:
				data.append(data_set)
			i += 1
		filename.close()

		N = len(data)
		self.x_eta = np.zeros(N)
		self.Etas = np.zeros(N)
		for j in range(N):
			self.x_eta[j] = float(data[j][0])
			self.Etas[j] = float(data[j][1])

	def Plot_data(self):
		self.read_data("Eta_test.txt")
		plt.figure()
		plt.semilogy(self.x_eta, self.Etas/(3.08568025e25))
		plt.hold("on")
		self.read_data("InterpTest.txt")
		plt.semilogy(self.x_eta, self.Etas/(3.08568025e25), 'rx')
		plt.show()

solver = Plotter(True)
solver.Plot_data()