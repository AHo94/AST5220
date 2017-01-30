import numpy as np
import matplotlib.pyplot as plt

# Global constants
# Units
eV = 1.60217647e-19
Mpc = 3.08568025e22

# Cosmological parameters
Omega_b = 0.046
Omega_m = 0.224
Omega_r = 8.3e-5
Omega_nu = 0.0
Omega_lambda = 1.0 - Omega_m - Omega_b - Omega_r - Omega_nu
T_0 = 2.725
n_s = 1.0
A_s = 1.0
h0 = 0.7
H_0 = h0*100.0*1e3/Mpc

# General constants
c = 2.99792458e8
epsilon_0 = 13.605698*eV
m_e = 9.10938188e-31
m_H = 1.673534e-27
sigma_T = 6.652462e-29
G_grav = 6.67258e-11
rho_c = (3.0*H_0**2)/(8*np.pi*G_grav)
alpha = 7.29735308e-3
hbar = 1.05457148e-34
k_b = 1.3806503e-23


class time_mod():
	def __init__(self, h):
		self.h = h 					# Step size for the RK4 method
		self.RK_Prefactor = (h/6.0)	# Precalculating prefactor of RK4 method to reduce float point operations

		self.n1 = 200
		self.n2 = 300
		self.n_t = self.n1 + self.n2

		self.z_start_rec = 1630.4
		self.z_end_rec = 614.2
		self.z_0 = 0.0
		self.x_start_rec = -np.log(1.0 + self.z_start_rec)
		self.x_end_rec = -np.log(1.0 + self.z_end_rec)
		self.x_0 = 0.0
		self.a_start_rec = 1.0/(1.0 + self.z_start_rec)
		self.a_end_rec = 1.0/(1.0 + self.z_end_rec)

		self.n_eta = 1000
		self.a_init = 1e-10
		self.x_eta_init = np.log(self.a_init)
		self.x_eta_end = 0

		# Set up grid
		x_t_rec = np.linspace(self.x_start_rec, self.x_end_rec, self.n1)
		x_t_today = np.linspace(self.x_end_rec, self.x_0, self.n2)
		a_t_rec = np.linspace(self.a_start_rec, self.a_end_rec, self.n1)
		a_t_today = np.linspace(self.a_end_rec, 1, self.n2)
		# Merging the arrays into one
		self.x_t = np.concatenate([x_t_rec, x_t_today])
		self.a_t = np.concatenate([a_t_rec, a_t_today])
		self.x_eta = np.linspace(self.x_eta_init, self.x_eta_end, self.n_t)	# X-values for the conformal time

		# Set up array for the calculated variables
		self.eta_array = np.zeros(self.n_t)

	def Get_Hubble_param(self, x):
		""" Function returns the Hubble parameter for a given x """
		return H_0*np.sqrt((Omega_b + Omega_m)*np.exp(-3*x) + Omega_r*np.exp(-4*x) + Omega_lambda)

	def Get_Hubble_prime(self, x):
		""" Function returns the scaled Hubble parameter for a given x value. See report """
		return H_0*np.sqrt((Omega_b + Omega_m)*np.exp(-x) + Omega_r*np.exp(-2*x) + Omega_lambda)

	def Get_Hubble_prime_derivative(self, x):
		""" Function returns the derivative of the scaled Hubble parameter. See report """
		return -H_0**2*((Omega_b + Omega_m)*np.exp(-x) + 2*Omega_r*np.exp(-2*x))/(2*Get_Hubble_prime(x))

	def RungeKutta4(self, i):
		""" Function solving the fourth order Runge-Kutta method """
		k1 = c/self.Get_Hubble_prime(self.x_eta[i])
		k2 = c/self.Get_Hubble_prime(self.x_eta[i] + self.h/2.0)
		k3 = c/self.Get_Hubble_prime(self.x_eta[i] + self.h/2.0)
		k4 = c/self.Get_Hubble_prime(self.x_eta[i] + self.h)

		return self.RK_Prefactor*(k1 + 2*k2 + 2*k3 + k4)

	def Spline(self, eta, x, interval_start, interval_end):
		""" Cubic spline interpolation. From Numerical recipies """
		interpolated_points = np.zeros(interval_end - interval_start)
		self.eta_d2 = self.Spline_second_derivative(eta, x, interval_start, interval_end)
		for i in range(interval_start, interval_end):
			"""
			A = (x[i+1]-x)/(x[i+1] - x[i])
			B = 1 - A
			CD_factor = (1.0/6.0)*(x[i+1] - x[i])**2
			C = CD_factor*(A**3 - A)
			D = CD_factor*(B**3 - B)
			"""
			#interpolated_points[i] = A*eta[i] + B*eta[i+1] + C*eta_d2[i] + D*eta_d2[i+1]

	def Spline_derivative(self):
		a = 1

	def Spline_second_derivative(self, y, x, index1, index2):
		""" 
		Second derivative of the spline interpolation.
		Using natural cubic spline, i.e. zero double derivatives at the boundaries
		"""
		y_doublederivative = np.zeros(self.n_t)
		for i in range(0, 10):
			y_doublederivative[i+1] = (6.0/(x[i+1] - x[i]))*((y[i+1] - y[i])/(x[i+1] - x[i])\
			 - (y[i] - y[i-1])/(x[i] - x[i-1]) - (x[i+1] - x[i-1])*y_doublederivative[i]/3.0\
			 - (x[i] - x[i-1])*y_doublederivative[i-1]/6.0)
		print y_doublederivative
		return y_doublederivative

	def Solve_Comformal_time(self):
		""" Solving the differential equation """
		for i in range(1, self.n_t-1):
			self.eta_array[i+1] = self.eta_array[i] + self.RungeKutta4(i)

	def Plot_results(self):
		""" Plotting the results """
		self.Solve_Comformal_time()
		self.Spline(self.eta_array, self.x_t, 0, self.n_t)
		#print self.x_t
		#print self.eta_d2

		#plt.plot(self.x_t, self.eta_array)
		#plt.show()

solver = time_mod(1e-3)
solver.Plot_results()