import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
# Change directory based based on work place, i.e home or at UiO computer.
file_directory = 'C:/Users/Alex/Documents/AST5220/AST5220_Projects/Project3/Project3_cpp/build-Project3-Desktop_Qt_5_7_0_MinGW_32bit-Debug'

class Plotter():
	def __init__(self, savefile):
		self.savefile = savefile	# If 1, saves to png file. Shows the plot otherwise

	def read_data(self, filename_open):
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
		self.x_values = np.zeros(N)
		self.Theta0 = np.zeros(N)
		self.Theta1 = np.zeros(N)
		self.Theta2 = np.zeros(N)
		self.Theta3 = np.zeros(N)
		self.Theta4 = np.zeros(N)
		self.Theta5 = np.zeros(N)
		self.Theta6 = np.zeros(N)
		self.delta = np.zeros(N)
		self.deltab = np.zeros(N)
		self.v = np.zeros(N)
		self.vb = np.zeros(N)
		self.Phi = np.zeros(N)
		for j in range(N):
			self.x_values[j] = float(data[j][0])
			self.Theta0[j] = float(data[j][1])
			self.Theta1[j] = float(data[j][2])
			self.Theta2[j] = float(data[j][3])
			self.Theta3[j] = float(data[j][4])
			self.Theta4[j] = float(data[j][5])
			self.Theta5[j] = float(data[j][6])
			self.Theta6[j] = float(data[j][7])
			self.delta[j] = float(data[j][8])
			self.deltab[j] = float(data[j][9])
			self.v[j] = float(data[j][10])
			self.vb[j] = float(data[j][11])
			self.Phi[j] = float(data[j][12])

	def Plot_data(self):
		ks = np.linspace(0, 99, 100)
		#for k_value in ks:
		#	print str(k_value)

		self.read_data("TEST.txt")
		plt.figure()
		plt.plot(self.x_values, self.Theta0)
		plt.xlabel('x')
		plt.ylabel('$\Theta_0$')

		plt.figure()
		plt.plot(self.x_values, self.Phi)
		plt.xlabel('x')
		plt.ylabel('$\Phi$')

		if self.savefile == 1:
			a = 1
		else:
			plt.show()


solver = Plotter(0)
solver.Plot_data()