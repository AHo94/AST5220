import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate

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
	def __init__(self, savefig):
		self.savefig = savefig		# If savefig = 0, plots the data. If savefig = 1, saves the plots in a pdf
		if savefig != 0 and savefig != 1:
			print 'Current value of savefig = ', savefig
			raise ValueError('Argument savefig not properly set. Try savefig = 1 (saves as pdf) or savefig = 0 (do not save as pdf)')
			
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

		# Used for the x-values for the conformal time
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
		self.x_eta = np.linspace(self.x_eta_init, self.x_eta_end, self.n_eta)	# X-values for the conformal time
		self.eta_array = np.zeros(self.n_eta)

	def Get_Hubble_param(self, x):
		""" Function returns the Hubble parameter for a given x """
		return H_0*np.sqrt((Omega_b + Omega_m)*np.exp(-3*x) + Omega_r*np.exp(-4*x) + Omega_lambda)

	def Get_Hubble_prime(self, x):
		""" Function returns the scaled Hubble parameter for a given x value. See report """
		return H_0*np.sqrt((Omega_b + Omega_m)*np.exp(-x) + Omega_r*np.exp(-2*x) + Omega_lambda*np.exp(2*x))

	def Get_Hubble_prime_derivative(self, x):
		""" Function returns the derivative of the scaled Hubble parameter. See report """
		return -H_0**2*((Omega_b + Omega_m)*np.exp(-x) + 2*Omega_r*np.exp(-2*x))/(2*Get_Hubble_prime(x))

	def Diff_eq(self, y, x_init):
		""" Returns the right hand side of the differential equation """
		dEtada = c/(self.Get_Hubble_prime(x_init))
		return dEtada

	def Get_eta(self, x_values, eta_values, x_start, x_end, n_points):
		""" Cubic spline interpolation, zeroth derivative. Returns interpolated eta for a given range of x-values """
		Temp_interp = interpolate.splrep(x_values, eta_values)
		x_new = np.linspace(x_start, x_end, n_points)
		eta_new = interpolate.splev(x_new, Temp_interp, der=0)
		return x_new, eta_new

	def Spline(self, x_values, eta_values):
		""" 
		Evaluates the second derivatives at each grid point.
		Boundaries for the double derivatives are zero, using the so called natural spline 
		"""
		Temp_interp = interpolate.splrep(x_values, eta_values)
		etaDoubleDer = interpolate.splev(x_values, Temp_interp, der=2)
		etaDoubleDer[0] = 0
		etaDoubleDer[-1] = 0
		return etaDoubleDer

	def Plot_results(self, n_interp_points):
		""" Solves and plots the results """
		self.ScipyEta = integrate.odeint(self.Diff_eq, self.x_eta_init, self.x_eta)
		#EtaDoubleDer = self.Spline(self.x_eta, self.ScipyEta)
		x_eta_new, eta_new = self.Get_eta(self.x_eta, self.ScipyEta, self.x_start_rec, self.x_end_rec, n_interp_points)

		fig1 = plt.figure()
		ax1 = plt.subplot(111)
		plt.hold("on")
		ax1.plot(self.x_eta, self.ScipyEta, 'b-', label='Scipy integrated')
		ax1.plot(x_eta_new, eta_new, 'xr', label='Interpolated')
		plt.xlabel('x')
		plt.ylabel('$\eta - [m]$')
		ax1.legend(loc='upper left', bbox_to_anchor=(0.5,1), ncol=1, fancybox=True)
		plt.title('Plot of conformal time $\eta$ as a function of $x = \ln a$')
		print self.ScipyEta[-1]/(3.0856*10**(16)*10**(9))
		
		fig2 = plt.figure()
		ax2 = plt.subplot(111)
		plt.hold("on")		
		ax2.plot(self.x_eta, self.ScipyEta, 'b-', label='Scipy integrated')
		ax2.plot(x_eta_new, eta_new, 'xr', label='Interpolated')
		EtaIndex1 = (np.abs(self.x_eta - self.x_start_rec)).argmin()
		EtaIndex2 = (np.abs(self.x_eta - self.x_end_rec)).argmin()
		plt.axis([self.x_start_rec-1, self.x_end_rec+1, self.ScipyEta[EtaIndex1], self.ScipyEta[EtaIndex2]])
		plt.legend(['Scipy integated','Interpolated'])
		plt.xlabel('x')
		plt.ylabel('$\eta - [m]$')
		ax2.legend(loc='upper right', bbox_to_anchor=(1,0.5), ncol=1, fancybox=True)
		plt.title('Plot of conformal time $\eta$ as a function of $x = \ln a$. \n Zoomed in the interpolated part.')

		fig3 = plt.figure()
		ax3 = plt.subplot(111)
		ax3.plot(self.x_eta, self.Get_Hubble_param(self.x_eta)*Mpc/1e3)
		plt.xlabel('x')
		plt.ylabel(r'$H - [km/s/Mpc]$')
		plt.title('Hubble parameter as a function of $x = \ln a$.')

		fig4 = plt.figure()
		ax4 = plt.subplot(111)
		ax4.plot(self.x_eta, self.ScipyEta/(Mpc*1e3), 'b-', label='Scipy integrated')
		ax4.plot(x_eta_new, eta_new/(Mpc*1e3), 'xr', label='Interpolated')
		plt.xlabel('x')
		plt.ylabel('$\eta - [Gpc]$')
		ax4.legend(loc='upper left', bbox_to_anchor=(0.5,1), ncol=1, fancybox=True)
		plt.title('Plot of conformal time $\eta$ as a function of $x = \ln a$. \n $\eta$ in units of Gpc')
		print self.Get_Hubble_param(self.x_eta[-1])*Mpc/1e3

		if self.savefig == 1:
			fig1.savefig('../Plots/Interpolated_Example.pdf')
			fig2.savefig('../Plots/Interpolated_Example_zoomed.pdf')
			fig3.savefig('../Plots/Hubble_parameter.pdf')
		else:
			plt.show()

solver = time_mod(0)
solver.Plot_results(100)
