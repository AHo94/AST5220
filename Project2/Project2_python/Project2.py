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
rho_c0 = (3.0*H_0**2)/(8*np.pi*G_grav)
alpha = 7.29735308e-3
hbar = 1.05457148e-34
k_b = 1.3806503e-23

# Density Parameters today
rho_m0 = Omega_m*rho_c0
rho_b0 = Omega_b*rho_c0
rho_r0 = Omega_r*rho_c0
rho_lambda0 = Omega_lambda*rho_c0

# Precalculate certain factors to reduce number of float point operations
Saha_b_factor = m_e*T_0/(2*np.pi)		# Factor in front of 'b' in Saha equation
rhoCrit_factor = 3.0/(8*np.pi*G_grav)	# Used for critical density at arbitrary times

# Constant used for Peebles equation and some constant factors that can be precalculated
Lambda_2sto1s = 8.227
alpha_factor = 64*np.pi/(np.sqrt(27)*np.pi)*(alpha*alpha/(m_e*m_e))*(k_b/(hbar*c))
beta_factor = (m_e*T_0/(2*np.pi))**(3.0/2.0)
Lambda_alpha_factor = (3*epsilon_0*c/hbar)**3/(8*np.pi)**2
EpsTemp_factor = epsilon_0/(k_b*T_0)
dXedx_factor = hbar**2/(np.sqrt(k_b)*c)

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

		# Set up grid, these are currently unused
		x_t_rec = np.linspace(self.x_start_rec, self.x_end_rec, self.n1)
		x_t_today = np.linspace(self.x_end_rec, self.x_0, self.n2)
		a_t_rec = np.linspace(self.a_start_rec, self.a_end_rec, self.n1)
		a_t_today = np.linspace(self.a_end_rec, 1, self.n2)
		# Merging the arrays into one
		self.x_t = np.concatenate([x_t_rec, x_t_today])
		self.a_t = np.concatenate([a_t_rec, a_t_today])

		# Set up grid of x-values for the integrated eta
		self.x_eta = np.linspace(self.x_eta_init, self.x_eta_end, self.n_eta)	# X-values for the conformal time
		self.x_tau = np.linspace(self.x_eta_end, self.x_eta_init, self.n_eta)

		self.X_e = self.Saha_equation(self.x_eta_init)
		#self.X_e = 0.99
		self.X_e_counter = 0

		# Solves the equations of Eta and interpolates
		#self.ScipyEta = integrate.odeint(self.Diff_eq_eta, self.x_eta_init, self.x_eta)
		#x_eta_new, eta_new = self.Get_eta(self.x_eta, self.ScipyEta, x_start, x_end, n_interp_points)
		#self.Saha_equation(self.x_eta)

	def Get_Hubble_param(self, x):
		""" Function returns the Hubble parameter for a given x """
		return H_0*np.sqrt((Omega_b + Omega_m)*np.exp(-3*x) + Omega_r*np.exp(-4*x) + Omega_lambda)

	def Get_Hubble_prime(self, x):
		""" Function returns the scaled Hubble parameter for a given x value. See report """
		return H_0*np.sqrt((Omega_b + Omega_m)*np.exp(-x) + Omega_r*np.exp(-2*x) + Omega_lambda*np.exp(2*x))

	def Get_Hubble_prime_derivative(self, x):
		""" Function returns the derivative of the scaled Hubble parameter. See report """
		return -H_0**2*(0.5*(Omega_b + Omega_m)*np.exp(-x) + Omega_r*np.exp(-2*x) - Omega_lambda*np.exp(2*x))/(Get_Hubble_prime(x))

	def Get_Omegas(self, x):
		""" 
		Calculates the omegas as a function of redshift
		Will first have to calculate the energy densities today, which is then used to calculate the energy density
		for an arbitrary time. See report
		"""
		H = self.Get_Hubble_param(x)		# Hubble parameter for an arbitrary time
		rho_c = rhoCrit_factor*H**2			# Critical density for an arbitrary time
		Omega_m_z = rho_m0*np.exp(-3*x)/rho_c
		Omega_b_z = rho_b0*np.exp(-3*x)/rho_c
		Omega_r_z = rho_r0*np.exp(-4*x)/rho_c
		Omega_lambda_z = rho_lambda0/rho_c

		return Omega_m_z, Omega_b_z, Omega_r_z, Omega_lambda_z

	def Diff_eq_eta(self, eta, x_0):
		""" Returns the right hand side of the differential equation for the conformal time eta """
		dEtada = c/(self.Get_Hubble_prime(x_0))
		return dEtada

	def Get_eta(self, x_values, eta_values, x_start, x_end, n_points):
		""" Cubic spline interpolation, zeroth derivative. Returns interpolated eta for a given range of x-values """
		Temp_interp = interpolate.splrep(x_values, eta_values)
		x_new = np.linspace(x_start, x_end, n_points)
		eta_new = interpolate.splev(x_new, Temp_interp, der=0)
		return x_new, eta_new

	def Spline_DoubleDerivative(self, x_values, eta_values):
		""" 
		Evaluates the second derivatives at each grid point.
		Boundaries for the double derivatives are zero, using the so called natural spline 
		"""
		Temp_interp = interpolate.splrep(x_values, eta_values)
		etaDoubleDer = interpolate.splev(x_values, Temp_interp, der=2)
		etaDoubleDer[0] = 0
		etaDoubleDer[-1] = 0
		return etaDoubleDer

	def Get_Index_Interpolation(self, X_init, X_end):
		""" 
		Finds the array index/component of x for a given x-value
		This is specifically used to zoom into the interpolated segment
		"""
		EtaIndex1 = (np.abs(self.x_eta - X_init)).argmin()
		EtaIndex2 = (np.abs(self.x_eta - X_end)).argmin()
		if EtaIndex1-1 <= 0:
			EtaIndex1 = 0
		else:
			EtaIndex1 -= 1

		if EtaIndex2+1 >= self.n_eta:
			EtaIndex2 = self.n_eta-1
		else:
			EtaIndex2 += 1

		return EtaIndex1, EtaIndex2

	def Get_n_b(self, x):
		""" Calculate n_b (or n_H) at a given time """
		Om_m, Om_b, Om_r, Om_lamda = self.Get_Omegas(x)
		H = self.Get_Hubble_param(x)		
		rho_c = rhoCrit_factor*H**2	
		n_b = Om_b*rho_c0/(m_H*np.exp(3*x))
		return n_b

	def Saha_equation(self, x):
		""" 
		Solves the Saha equation. Assuming we have the polynomial in the form a*X_e^2 + b*X_e + c = 0
		Uses numpy.roots solver. Only returns the positive valued X_e 
		"""
		Exponential = np.exp(x)
		a = self.Get_n_b(x)
		b = (Saha_b_factor/Exponential)**(3.0/2.0)*np.exp(-EpsTemp_factor*Exponential)
		c = -b
		X_e = np.roots(np.array([a,b,c]))

		if X_e[0] > 0:
			return X_e[0]
		else:
			return X_e[1]

	def Peebles_equation(self, X_e, x_0):
		""" Solves the right hand side of the Peebles equation """
		n_b = self.Get_n_b(x_0)
		H = self.Get_Hubble_param(x_0)
		exp_factor = EpsTemp_factor*np.exp(x_0)
		phi2 = 0.448*np.log(exp_factor)
		alpha2 = alpha_factor*np.sqrt(exp_factor)*phi2
		beta = alpha2*beta_factor*np.exp(-EpsTemp_factor)
		beta2 = beta*np.exp(3.0*exp_factor/4.0)
		Lambda_alpha = H*Lambda_alpha_factor/((1-X_e)*n_b)
		C_r = (Lambda_2sto1s + Lambda_alpha)/(Lambda_2sto1s + Lambda_alpha + beta2)
		dXedx = (C_r/H)*(beta*(1-X_e) - dXedx_factor*n_b*alpha2*X_e**2)
		return dXedx


	def Diff_eq_tau(self, tau, x_0):
		""" 
		Solves the differential equation of tau. This is the right hand side of the equation
		Uses Saha equation if X_e is roughly 1, else uses Peebles equation
		"""
		n_b = self.Get_n_b(x_0)
		#self.X_e = self.Saha_equation(x_0)
		#self.X_e = integrate.odeint(self.Peebles_equation, self.X_e, x_0)[0][0]
		if self.X_e_counter == 1:
			if self.X_e > 1e-1:
				self.X_e = self.Saha_equation(x_0)
			else:
				self.X_e = integrate.odeint(self.Peebles_equation, self.X_e, x_0)[0][0]
		n_e = self.X_e*n_b
		dTaudx = - n_e*sigma_T/self.Get_Hubble_param(x_0)
		self.X_e_counter = 1
		return dTaudx

	def Plot_results(self, n_interp_points, x_start = -np.log(1.0 + 1630.4), x_end = -np.log(1.0 + 614.2)):
		""" Solves and plots the results """
		self.ScipyEta = integrate.odeint(self.Diff_eq_eta, 0, self.x_eta)
		#x_eta_new, eta_new = self.Get_eta(self.x_eta, self.ScipyEta, x_start, x_end, n_interp_points)
		Taus = integrate.odeint(self.Diff_eq_tau, 0, self.x_tau)
		plt.semilogy(self.x_tau, Taus)
		plt.xlabel('x')
		plt.ylabel(r'$\tau$')
		plt.show()

		if self.savefig == 1:
			a = 1
		else:
			plt.show()

solver = time_mod(savefig=0)
#solver.Saha_equation()
solver.Plot_results(100)
