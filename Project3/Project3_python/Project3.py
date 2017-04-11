import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import integrate
import time
import cProfile

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
Saha_b_factor = ((m_e*T_0*k_b)/(2*np.pi*hbar**2))**(3.0/2.0)	# Factor in front of 'b' in Saha equation
rhoCrit_factor = 3.0/(8*np.pi*G_grav)							# Used for critical density at arbitrary times

# Constant used for Peebles equation and some constant factors that can be precalculated
Lambda_2sto1s = 8.227
alpha_factor = ((64.0*np.pi)/(np.sqrt(27.0*np.pi)))*((alpha/m_e)**2.0)*(hbar**2.0/c)
beta_factor = (((m_e*T_0*k_b)/(2.0*np.pi))**(3.0/2.0))*(1.0/hbar**3.0)
Lambda_alpha_factor = ((3.0*epsilon_0/(hbar*c))**3.0)/(8*np.pi)**2.0
EpsTemp_factor = epsilon_0/(k_b*T_0)

# Other precalculated factors
H_0Squared = H_0*H_0
c_Squared = c*c
PsiPrefactor = 12.0*H_0*H_0/(c*c)

class time_mod():
	def __init__(self, savefig, l_max):
		self.savefig = savefig		# If savefig = 0, plots the data. If savefig = 1, saves the plots into a pdf

		if savefig != 0 and savefig != 1:
			print 'Current value of savefig = ', savefig
			raise ValueError('Argument savefig not properly set. Try savefig = 1 (saves as pdf) or savefig = 0 (do not save as pdf)')

		self.time_start = time.clock()
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
		self.n_eta = 3000
		self.a_init = 1e-8
		self.x_eta_init = np.log(self.a_init)
		self.x_eta_end = 0

		# Set up grid, these are currently unused
		self.x_t_rec = np.linspace(self.x_start_rec, self.x_end_rec, self.n1)
		self.x_t_today = np.linspace(self.x_end_rec, self.x_0, self.n2)
		# Merging the arrays into one
		self.x_t = np.concatenate([self.x_t_rec, self.x_t_today])

		# Set up grid of x-values for the integrated eta
		self.x_eta = np.linspace(self.x_eta_init, self.x_eta_end, self.n_eta)	# X-values for the conformal time
		self.x_tau = np.linspace(self.x_eta_end, self.x_eta_init, self.n_eta)	# Reversed array, used to calculate tau

		self.l_max = l_max
		self.lValues = np.linspace(2, l_max-1, l_max-2)
		self.NumVariables = self.l_max + 1 + 5
		k_min = 0.1*H_0
		k_max = 340*H_0
		self.k_N = 2
		#self.k1 = np.linspace(k_min, k_max, self.k_N)
		self.k = np.array([k_min + (k_max-k_min)*(i/100.0)**2 for i in range(self.k_N)])
		#print self.k-self.k1
		self.k_squared = self.k*self.k
		self.ck = c*self.k

		self.hmin = (self.x_t_today[-1]-self.x_t_today[0])/float(self.n2)
		"""
		k = 1 , 16 seconds
		k = 2 , 24 seconds, + 8 s
		k = 3 , 42 seconds, + 18 s
		k = 4 , 55 seconds, + 13 s
		"""

		Ks = np.array([1,2])
		Ls = np.array([0,1,2,3,4])
		B = np.array([1,2,3,4,5])
		A = np.array([0,0,0,0,0])
		AA = np.array([[1,2],[2,3]])
		#print Ls[1:-1]
		#A[1:-1] = Ls[1:-1]*B[2:] + B[0:-2]
		
		# LL = (l*k)
		LL = np.array([1*np.array([1.0,2.0,3.0,4.0]), 2*np.array([1.0,2.0,3.0,4.0]), 3*np.array([1.0,2.0,3.0, 4.0])])
		LLdiv = np.array([1.0,2.0,3.0])
		THET = np.array([10,20,30])
		THET0 = np.zeros((3,4))
		print THET0
		THET0[1:-1] = LL[1:-1]/(2.0*LLdiv[1:-1]+1)*THET[0]
		print THET0

		"""
		Premake l*k array, so it is a (6x100) array
		Calculate thetas as above, index every array possible
		first element array corresponds to the l values, each l value multiplied with thetas applies for all k 
		"""

	def Get_Hubble_param(self, x):
		""" Function returns the Hubble parameter for a given x """
		return H_0*np.sqrt((Omega_b + Omega_m)*np.exp(-3*x) + Omega_r*np.exp(-4*x) + Omega_lambda)

	def Get_Hubble_prime(self, x):
		""" Function returns the scaled Hubble parameter for a given x value. See report 1 """
		return H_0*np.sqrt((Omega_b + Omega_m)*np.exp(-x) + Omega_r*np.exp(-2*x) + Omega_lambda*np.exp(2*x))

	def Get_Hubble_prime_derivative(self, x):
		""" Function returns the derivative of the scaled Hubble parameter. See report 1 """
		return -H_0**2*(0.5*(Omega_b + Omega_m)*np.exp(-x) + Omega_r*np.exp(-2*x) - Omega_lambda*np.exp(2*x))/(self.Get_Hubble_prime(x))

	def Get_Omegas(self, x):
		""" 
		Calculates the omegas as a function of redshift
		Will first have to calculate the energy densities today, which is then used to calculate the energy density
		for an arbitrary time. See report 1
		"""
		H = self.Get_Hubble_param(x)
		rho_c = rhoCrit_factor*H**2
		Omega_m_z = rho_m0*np.exp(-3*x)/rho_c
		Omega_b_z = rho_b0*np.exp(-3*x)/rho_c
		Omega_r_z = rho_r0*np.exp(-4*x)/rho_c
		Omega_lambda_z = rho_lambda0/rho_c

		return Omega_m_z, Omega_b_z, Omega_r_z, Omega_lambda_z

	def Diff_eq_eta(self, eta, x_0):
		""" Returns the right hand side of the differential equation for the conformal time eta """
		dEtada = c/(self.Get_Hubble_prime(x_0))
		return dEtada

	def Cubic_Spline(self, x_values, y_values, n_points, x_start=np.log(1e-8), x_end=0):
		""" 
		Cubic spline interpolation, zeroth derivative. Returns interpolated values of any variables, for a given range of x-values
		"""
		Temp_interp = interpolate.splrep(x_values, y_values)
		x_new = np.linspace(x_start, x_end, n_points)
		y_new = interpolate.splev(x_new, Temp_interp, der=0)
		return x_new, y_new

	def Cubic_Spline_OnePoint(self, x_values, y_values, x_point):
		""" Cubic spline for one specific point """
		Temp_interp = interpolate.splrep(x_values, y_values)
		y_new = interpolate.splev(x_point, Temp_interp, der=0)
		return y_new

	def Spline_Derivative(self, x_values, y_values, n_points, derivative, x_start=np.log(1e-8), x_end=0):
		""" Spline derivative for any functions. Using natural spline for the second derivative """
		if derivative < 1:
			raise ValueError("Derivative input in Spline_Derivative less than 1. Use Cubic_spline instead.")
		Temp_interp = interpolate.splrep(x_values, y_values)
		x_new = np.linspace(x_start, x_end, n_points)
		yDerivative = interpolate.splev(x_new, Temp_interp, der=derivative)
		if derivative == 2:
			yDerivative[0] = 0
			yDerivative[-1] = 0
		return yDerivative

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
		""" Calculate n_b (or n_H) at a given 'time' x """
		n_b = Omega_b*rho_c0*np.exp(-3.0*x)/m_H
		return n_b

	def Saha_equation(self, x):
		""" Solves the Saha equation. Uses numpy.roots solver, see report 2. Only returns the positive valued X_e """
		Exponential = np.exp(x)
		a = 1
		b = (Saha_b_factor/self.Get_n_b(x))*np.exp(-EpsTemp_factor*Exponential - 3.0*x/2.0)
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
		beta = alpha2*beta_factor*np.exp(-3.0*x_0/2.0-exp_factor)
		beta2 = alpha2*beta_factor*np.exp(-3.0*x_0/2.0-exp_factor/4.0)
		Lambda_alpha = H*Lambda_alpha_factor/((1.0-X_e)*n_b)
		C_r = (Lambda_2sto1s + Lambda_alpha)/(Lambda_2sto1s + Lambda_alpha + beta2)
		dXedx = (C_r/H)*(beta*(1.0-X_e) - n_b*alpha2*X_e**2.0)
		return dXedx

	def Calculate_Xe(self):
		""" Function that calculates X_e. Initial condition X_e = 1 """
		X_e_TempArray = [1]
		Peeble = False
		for i in range(0,self.n_eta-1):
			if X_e_TempArray[i] > 0.99:
				X_e_TempArray.append(self.Saha_equation(self.x_eta[i]))
			else:
				PeebleXe = integrate.odeint(self.Peebles_equation, X_e_TempArray[i], self.x_eta[i:])
				break
		PeebleXe2 = []
		for i in range(0, len(PeebleXe)-1):
			PeebleXe2.append(PeebleXe[i][0])
		self.X_e_array = np.concatenate([np.array(X_e_TempArray),np.array(PeebleXe2)])

	def Diff_eq_tau(self, tau, x_0):
		""" 
		Solves the differential equation of tau. This is the right hand side of the equation
		Finds the n_e value that corresponds to the x value, since we use a reversed x-array.
		"""
		n_e = np.exp(self.Cubic_Spline_OnePoint(self.x_eta, np.log(self.n_e), x_0))
		dTaudx = -n_e*sigma_T*c/self.Get_Hubble_param(x_0)
		return dTaudx

	def Visibility_func(self, x, tau, tauDerv):
		""" Computes the visibility function (tilde) """
		g = np.zeros(len(tau))
		for i in range(0, len(tau)-1):
			g[i] = -tauDerv[i]*np.exp(-tau[i])
		return g

	def Kronecker_Delta_2(self, l):
		""" Kronecker delta that only returns 1 if l = 2 """
		if l == 2:
			return 1
		else:
			return 0

	def BoltzmannEinstein_InitConditions(self):
		""" Initial conditions for the Boltzmann equations """
		Phi = 1.0*np.ones(self.k_N)
		delta_b = 3.0*Phi/2.0
		HPrime_0 = self.Get_Hubble_param(self.x_eta_init)
		InterpolateTauDerivative = self.Spline_Derivative(self.x_eta, self.Taus, 1, derivative = 1, x_start = self.x_eta_init, x_end =self.x_eta_init)
		v_b = self.ck*Phi/(2.0*HPrime_0)
		Theta_0 = 0.5*Phi
		Theta_1 = -c*self.k*Phi/(6.0*HPrime_0)
		Theta_2 = -8.0*c*self.k*Theta_1/(15*InterpolateTauDerivative*HPrime_0)
		"""
		self.BoltzmannVariables = []
		self.BoltzmannVariables.append(Theta_0)
		self.BoltzmannVariables.append(Theta_1)
		self.BoltzmannVariables.append(Theta_2)
		if self.l_max > 2:
			for l in range(3, self.l_max+1):
				self.BoltzmannVariables.append(-(l/(2.0*l+1))*(c*self.k/(HPrime_0*InterpolateTauDerivative))*self.BoltzmannVariables[l-1])
		else:
			raise ValueError('Value of l_max is a little too small. Try to increase it to l_max=3 or larger')

		self.BoltzmannVariables.append(delta_b)
		self.BoltzmannVariables.append(delta_b)
		self.BoltzmannVariables.append(v_b)
		self.BoltzmannVariables.append(v_b)
		self.BoltzmannVariables.append(Phi)
		self.NumVariables = len(self.BoltzmannVariables)
		"""
		self.BoltzmannTightCoupling = np.array([Theta_0, Theta_1, delta_b, delta_b, v_b, v_b, Phi])
		self.NumVarTightCoupling = len(self.BoltzmannTightCoupling)

	def BoltzmannEinstein_InitConditions_AfterTC(self):
		""" 
		Properly set up all variables into a parameter in the tight coupling regime
		Also sets up initial conditions of the different parameters, that is to be calculated for time after recombination
		"""
		Transposed = np.transpose(self.EBTightCoupling)
		Hprimed = self.Get_Hubble_prime(self.x_t_rec)
		TauDer = self.Spline_Derivative(self.x_eta, self.Taus, self.n1, derivative=1, x_start=self.x_t_rec[0], x_end=self.x_t_rec[-1])
		Theta0 = []
		Theta1 = []
		Theta2 = []
		Theta3 = []
		Theta4 = []
		Theta5 = []
		Theta6 = []
		delta = []
		deltab = []
		v = []
		vb = []
		Phi = []
		for i in range(self.k_N):
			Theta0.append(Transposed[i])
			Theta1.append(Transposed[self.k_N+i])
			delta.append(Transposed[2*self.k_N+i])
			deltab.append(Transposed[3*self.k_N+i])
			v.append(Transposed[4*self.k_N+i])
			vb.append(Transposed[5*self.k_N+i])
			Phi.append(Transposed[6*self.k_N+i])
			Theta2.append(-20.0*c*self.k[i]*Theta1[i]/(45.0*Hprimed*TauDer))
			Theta3.append(-3.0*c*self.k[i]*Theta2[i]/(7.0*Hprimed*TauDer))
			Theta4.append(-4.0*c*self.k[i]*Theta3[i]/(9.0*Hprimed*TauDer))
			Theta5.append(-5.0*c*self.k[i]*Theta4[i]/(11.0*Hprimed*TauDer))
			Theta6.append(-6.0*c*self.k[i]*Theta5[i]/(13.0*Hprimed*TauDer))

		self.BoltzmannVariablesAFTERTC = [0]*self.NumVariables*self.k_N
		for j in range(self.k_N):
			self.BoltzmannVariablesAFTERTC[j] = Theta0[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N] = Theta1[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*2] = Theta2[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*3] = Theta3[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*4] = Theta4[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*5] = Theta5[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*6] = Theta6[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*7] = delta[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*8] = deltab[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*9] = v[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*10] = vb[j]
			self.BoltzmannVariablesAFTERTC[j+self.k_N*11] = Phi[j]
		
		self.BoltzmannVariablesAFTERTC_INIT = []
		for i in range(self.NumVariables*self.k_N):
			self.BoltzmannVariablesAFTERTC_INIT.append(self.BoltzmannVariablesAFTERTC[i][-1])

	def BoltzmannEinstein_Equations(self, variables, x_0):
		""" Solves Boltzmann Einstein equations """
		Theta_0, Theta_1, Theta_2, Theta_3, Theta_4, Theta_5, Theta_6, delta, delta_b, v, v_b, Phi = np.reshape(variables, (self.NumVariables, self.k_N))
		Om_m, Om_b, Om_r, Om_lamda = self.Get_Omegas(x_0)
		
		# Calculating some prefactors
		Hprimed = self.Get_Hubble_prime(x_0)
		Hprimed_Squared = Hprimed*Hprimed
		ck_Hprimed = self.ck/Hprimed
		# Interpolating Conformal time and Optical depth at the point x_0
		InterTauDerivative = self.Spline_Derivative(self.x_eta, self.Taus, 1, derivative=1, x_start=x_0, x_end=x_0)
		InterEta = self.Cubic_Spline_OnePoint(self.x_eta, self.ScipyEta, x_0)

		R = 4.0*Om_r/(3.0*Om_b*np.exp(x_0))
		Psi = -Phi - PsiPrefactor*(np.exp(-2.0*x_0)/(self.k_squared))*Om_r*Theta_2

		dPhidx = Psi - (ck_Hprimed**2/3.0)*Phi\
				+ (H_0Squared/(2.0*Hprimed_Squared))*(Om_m*np.exp(-x_0)*delta + Om_b*np.exp(-x_0)*delta_b + 4.0*Om_r*np.exp(-2.0*x_0)*Theta_0)

		ThetaDerivatives = []
		Thetas = [Theta_0, Theta_1, Theta_2, Theta_3, Theta_4, Theta_5, Theta_6]
		dTheta0dx = -ck_Hprimed*Theta_1 - dPhidx
		dTheta1dx = (ck_Hprimed/3.0)*Theta_0 - ((2.0*ck_Hprimed)/3.0)*Theta_2 \
					+ (ck_Hprimed/3.0)*Psi + InterTauDerivative*(Theta_1 + 1.0/(3.0*v_b))
		ThetaDerivatives.append(dTheta0dx)
		ThetaDerivatives.append(dTheta1dx)
		for l in range(2, self.l_max):
			dThetaldx = l*ck_Hprimed/(2.0*l+1.0)*Thetas[l-1] - ck_Hprimed*((l+1.0)/(2.0*l+1.0))*Thetas[l+1] \
						+ InterTauDerivative*(Thetas[l] - 0.1*Thetas[l]*self.Kronecker_Delta_2(l))
			ThetaDerivatives.append(dThetaldx)
		
		dThetalmaxdx = ck_Hprimed*Thetas[self.l_max-1] - c*((self.l_max + 1)/(Hprimed*InterEta))*Thetas[self.l_max]\
						+ InterTauDerivative*Thetas[self.l_max]

		ThetaDerivatives.append(dThetalmaxdx)
		dDeltadx = ck_Hprimed*v - 3.0*dPhidx
		dDeltabdx = ck_Hprimed*v_b - 3.0*dPhidx
		dvdx = -v - ck_Hprimed*Psi
		dvbdx = -v_b - ck_Hprimed*Psi + InterTauDerivative*R*(3.0*Theta_1 + v_b)

		derivatives = np.array([ThetaDerivatives[0], ThetaDerivatives[1], ThetaDerivatives[2], ThetaDerivatives[3], ThetaDerivatives[4] ,ThetaDerivatives[5]\
					, ThetaDerivatives[6], dDeltadx, dDeltabdx, dvdx, dvbdx, dPhidx])
		#print derivatives
		#print x_0
		return np.reshape(derivatives, self.NumVariables*self.k_N)

	def BoltzmannEinstein_Equations2(self, variables, x_0):
		""" Solves Boltzmann Einstein equations """
		Theta_0, Theta_1, Theta_2, Theta_3, Theta_4, Theta_5, Theta_6, delta, delta_b, v, v_b, Phi = np.reshape(variables, (self.NumVariables, self.k_N))
		Om_m, Om_b, Om_r, Om_lamda = self.Get_Omegas(x_0)
		
		# Calculating some prefactors
		Hprimed = self.Get_Hubble_prime(x_0)
		Hprimed_Squared = Hprimed*Hprimed
		ck_Hprimed = self.ck/Hprimed
		# Interpolating Conformal time and Optical depth at the point x_0
		InterTauDerivative = self.Spline_Derivative(self.x_eta, self.Taus, 1, derivative=1, x_start=x_0, x_end=x_0)
		InterEta = self.Cubic_Spline_OnePoint(self.x_eta, self.ScipyEta, x_0)

		R = 4.0*Om_r/(3.0*Om_b*np.exp(x_0))
		Psi = -Phi - PsiPrefactor*(np.exp(-2.0*x_0)/(self.k_squared))*Om_r*Theta_2

		dPhidx = Psi - (ck_Hprimed**2/3.0)*Phi\
				+ (H_0Squared/(2.0*Hprimed_Squared))*(Om_m*np.exp(-x_0)*delta + Om_b*np.exp(-x_0)*delta_b + 4.0*Om_r*np.exp(-2.0*x_0)*Theta_0)

		ThetaDerivatives = np.zeros((self.l_max, self.k_N))
		Thetas = np.array([Theta_0, Theta_1, Theta_2, Theta_3, Theta_4, Theta_5, Theta_6])
		ThetaDerivatives[0] = -ck_Hprimed*Theta_1 - dPhidx
		ThetaDerivatives[1] = (ck_Hprimed/3.0)*Theta_0 - ((2.0*ck_Hprimed)/3.0)*Theta_2 \
					+ (ck_Hprimed/3.0)*Psi + InterTauDerivative*(Theta_1 + 1.0/(3.0*v_b))
		ThetaDerivatives.append(dTheta0dx)
		ThetaDerivatives.append(dTheta1dx)

		for l in range(2, self.l_max):
			dThetaldx = l*ck_Hprimed/(2.0*l+1.0)*Thetas[l-1] - ck_Hprimed*((l+1.0)/(2.0*l+1.0))*Thetas[l+1] \
						+ InterTauDerivative*(Thetas[l] - 0.1*Thetas[l]*self.Kronecker_Delta_2(l))
			ThetaDerivatives.append(dThetaldx)
		
		dThetalmaxdx = ck_Hprimed*Thetas[self.l_max-1] - c*((self.l_max + 1)/(Hprimed*InterEta))*Thetas[self.l_max]\
						+ InterTauDerivative*Thetas[self.l_max]

		ThetaDerivatives.append(dThetalmaxdx)
		dDeltadx = ck_Hprimed*v - 3.0*dPhidx
		dDeltabdx = ck_Hprimed*v_b - 3.0*dPhidx
		dvdx = -v - ck_Hprimed*Psi
		dvbdx = -v_b - ck_Hprimed*Psi + InterTauDerivative*R*(3.0*Theta_1 + v_b)

		derivatives = np.array([ThetaDerivatives[0], ThetaDerivatives[1], ThetaDerivatives[2], ThetaDerivatives[3], ThetaDerivatives[4] ,ThetaDerivatives[5]\
					, ThetaDerivatives[6], dDeltadx, dDeltabdx, dvdx, dvbdx, dPhidx])
		#print derivatives
		#print x_0
		return np.reshape(derivatives, self.NumVariables*self.k_N)

	def TightCouplingRegime(self, variables, x_0):
		""" Boltzmann equation in the tight coupling regime """
		Theta_0, Theta_1, Theta_2, Theta_3, Theta_4, Theta_5, Theta_6, delta, delta_b, v, v_b, Phi = np.reshape(variables, (self.NumVariables, self.k_N))
		Om_m, Om_b, Om_r, Om_lamda = self.Get_Omegas(x_0)
		# Calculating some prefactors
		Hprimed = self.Get_Hubble_prime(x_0)
		HprimedDer = self.Get_Hubble_prime_derivative(x_0)
		#HPrimedDerivative = self.Get_Hubble_prime_derivative(x_0)
		Hprime_HprimeDer = Hprimed/HprimedDer
		Hprimed_Squared = Hprimed*Hprimed
		ck_Hprimed = self.ck/Hprimed
		# Interpolating Conformal time and Optical depth at the point x_0
		InterTauDerivative = self.Spline_Derivative(self.x_eta, self.Taus, 1, derivative=1, x_start=x_0, x_end=x_0)
		InterTauDoubleDer = self.Spline_Derivative(self.x_eta, self.Taus, 1, derivative=2, x_start=x_0, x_end=x_0)
		InterEta = self.Cubic_Spline_OnePoint(self.x_eta, self.ScipyEta, x_0)
		
		R = 4.0*Om_r/(3.0*Om_b*np.exp(x_0))
		Psi = -Phi - PsiPrefactor*(np.exp(-2.0*x_0)/(self.k_squared))*Om_r*Theta_2
		dPhidx = Psi - (ck_Hprimed**2/3.0)*Phi\
				+ (H_0Squared/(2.0*Hprimed_Squared))*(Om_m*np.exp(-x_0)*delta + Om_b*np.exp(-x_0)*delta_b + 4.0*Om_r*np.exp(-2.0*x_0)*Theta_0)
		dTheta0dx = -ck_Hprimed*Theta_1 - dPhidx
		q = -(((1.0-2.0*R)*InterTauDerivative + (1.0+R)*InterTauDoubleDer)*(3.0*Theta_1 + v_b) - ck_Hprimed*Psi \
					+ (1.0-Hprime_HprimeDer)*ck_Hprimed*(-Theta_0 + 2.0*Theta_2) - ck_Hprimed*dTheta0dx)/((1.0+R)*InterTauDerivative + Hprime_HprimeDer - 1.0)

		dDeltadx = ck_Hprimed*v - 3.0*dPhidx
		dDeltabdx = ck_Hprimed*v_b - 3.0*dPhidx
		dvdx = -v - ck_Hprimed*Psi
		dvbdx = (-v_b - ck_Hprimed*Psi + R*(q + ck_Hprimed*(-Theta_0 + 2.0*Theta_2) - ck_Hprimed*Psi))/(1.0+R)
		dTheta1dx = (q-dvbdx)/3.0
		# NEW STUFF
		HPTAU = Hprimed*InterTauDerivative
		dTheta2dx = (20.0*self.ck/(45.0))*(Theta_1*(HprimedDer/(Hprimed*HPTAU) + InterTauDoubleDer/(HPTAU*InterTauDerivative))+ dTheta1dx/HPTAU)
		dTheta3dx = (3.0/7.0)*self.ck*(Theta_2*(HprimedDer/(Hprimed*HPTAU) + InterTauDoubleDer/(HPTAU*InterTauDerivative) + dTheta2dx/(HPTAU)))
		dTheta4dx = (4.0/9.0)*self.ck*(Theta_3*(HprimedDer/(Hprimed*HPTAU) + InterTauDoubleDer/(HPTAU*InterTauDerivative) + dTheta3dx/(HPTAU)))
		dTheta5dx = (5.0/11.0)*self.ck*(Theta_4*(HprimedDer/(Hprimed*HPTAU) + InterTauDoubleDer/(HPTAU*InterTauDerivative) + dTheta4dx/(HPTAU)))
		dTheta6dx = (6.0/13.0)*self.ck*(Theta_5*(HprimedDer/(Hprimed*HPTAU) + InterTauDoubleDer/(HPTAU*InterTauDerivative) + dTheta5dx/(HPTAU)))
		derivatives = np.array([dTheta0dx, dTheta1dx, dTheta2dx, dTheta3dx, dTheta4dx, dTheta5dx, dTheta6dx, dDeltadx, dDeltabdx, dvdx, dvbdx, dPhidx])
		return np.reshape(derivatives, self.NumVariables*self.k_N)		

	def TightCouplingRegime2(self, variables, x_0):
		""" Boltzmann equation in the tight coupling regime """
		Theta_0, Theta_1, delta, delta_b, v, v_b, Phi = np.reshape(variables, (self.NumVarTightCoupling, self.k_N))
		Om_m, Om_b, Om_r, Om_lamda = self.Get_Omegas(x_0)
		# Calculating some prefactors
		Hprimed = self.Get_Hubble_prime(x_0)
		HprimedDer = self.Get_Hubble_prime_derivative(x_0)
		#HPrimedDerivative = self.Get_Hubble_prime_derivative(x_0)
		Hprime_HprimeDer = Hprimed/HprimedDer
		Hprimed_Squared = Hprimed*Hprimed
		ck_Hprimed = self.ck/Hprimed
		# Interpolating Conformal time and Optical depth at the point x_0
		InterTauDerivative = self.Spline_Derivative(self.x_eta, self.Taus, 1, derivative=1, x_start=x_0, x_end=x_0)
		InterTauDoubleDer = self.Spline_Derivative(self.x_eta, self.Taus, 1, derivative=2, x_start=x_0, x_end=x_0)
		InterEta = self.Cubic_Spline_OnePoint(self.x_eta, self.ScipyEta, x_0)
		
		Theta_2 = -20.0*ck_Hprimed*Theta_1/(45.0*InterTauDerivative)
		R = 4.0*Om_r/(3.0*Om_b*np.exp(x_0))
		Psi = -Phi - PsiPrefactor*(np.exp(-2.0*x_0)/(self.k_squared))*Om_r*Theta_2
		dPhidx = Psi - (ck_Hprimed**2/3.0)*Phi\
				+ (H_0Squared/(2.0*Hprimed_Squared))*(Om_m*np.exp(-x_0)*delta + Om_b*np.exp(-x_0)*delta_b + 4.0*Om_r*np.exp(-2.0*x_0)*Theta_0)
		dTheta0dx = -ck_Hprimed*Theta_1 - dPhidx
		q = -(((1.0-2.0*R)*InterTauDerivative + (1.0+R)*InterTauDoubleDer)*(3.0*Theta_1 + v_b) - ck_Hprimed*Psi \
					+ (1.0-Hprime_HprimeDer)*ck_Hprimed*(-Theta_0 + 2.0*Theta_2) - ck_Hprimed*dTheta0dx)/((1.0+R)*InterTauDerivative + Hprime_HprimeDer - 1.0)

		dDeltadx = ck_Hprimed*v - 3.0*dPhidx
		dDeltabdx = ck_Hprimed*v_b - 3.0*dPhidx
		dvdx = -v - ck_Hprimed*Psi
		dvbdx = (-v_b - ck_Hprimed*Psi + R*(q + ck_Hprimed*(-Theta_0 + 2.0*Theta_2) - ck_Hprimed*Psi))/(1.0+R)
		dTheta1dx = (q-dvbdx)/3.0
		derivatives = np.array([dTheta0dx, dTheta1dx, dDeltadx, dDeltabdx, dvdx, dvbdx, dPhidx])
		return np.reshape(derivatives, self.NumVarTightCoupling*self.k_N)		

	def Write_Outfile(self, filename, variables, k):
		""" Saves data to a text file """
		Transposed = variables#np.transpose(variables)
		#print Transposed[k+self.k_N*11]
		text_file = open(filename, "w")
		text_file.write(("Theta0, Theta1, Theta2, Theta3, Theta4, Theta5, Theta6, delta, delta_b, v, v_b, phi, k=%.8e \n") %self.k[k])
		for i in range(self.n_t):
			text_file.write(("%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e \n") \
			%(Transposed[k][i], Transposed[k+self.k_N][i], Transposed[k+self.k_N*2][i], Transposed[k+self.k_N*3][i],\
			 Transposed[k+self.k_N*4][i], Transposed[k+self.k_N*5][i], Transposed[k+self.k_N*6][i], Transposed[k+self.k_N*7][i],\
			 Transposed[k+self.k_N*8][i], Transposed[k+self.k_N*9][i], Transposed[k+self.k_N*10][i], Transposed[k+self.k_N*11][i]))
		text_file.close()

	def Plot_results(self, n_interp_points, x_start = -np.log(1.0 + 1630.4), x_end = -np.log(1.0 + 614.2)):
		""" Solves and plots the results """
		self.ScipyEta = integrate.odeint(self.Diff_eq_eta, 0, self.x_eta)
		# Calculate X_e, n_e and interpolates n_e as a test
		self.Calculate_Xe()
		self.n_e = self.X_e_array*self.Get_n_b(self.x_eta)
		x_eta_new, n_e_NewLogarithmic = self.Cubic_Spline(self.x_eta, np.log(self.n_e), n_interp_points)
		# Calculates tau and interpolates the first and second derivatives
		self.Taus = integrate.odeint(self.Diff_eq_tau, 0, self.x_tau)[::-1]	# Calculate tau and reverse array
		self.TauDerivative = self.Spline_Derivative(self.x_eta, self.Taus, self.n_eta, derivative=1)
		self.TauDoubleDer = self.Spline_Derivative(self.x_eta, self.Taus, 200, derivative=2)
		# Calculate g, and interpolates the first and second derivatives
		self.g_tilde = self.Visibility_func(self.x_eta, self.Taus, self.TauDerivative)
		self.g_tildeDerivative = self.Spline_Derivative(self.x_eta, self.g_tilde, self.n_eta, derivative=1)
		self.g_tildeDoubleDer = self.Spline_Derivative(self.x_eta, self.g_tilde, self.n_eta, derivative=2)
		print 'Setting Boltzmann initial conditions'
		self.BoltzmannEinstein_InitConditions()
		#print self.BoltzmannVariables
		print 'Calculating for tight coupling regime'
		#EBTightCoupling = integrate.odeint(self.TightCouplingRegime, np.reshape(self.BoltzmannVariables, self.NumVariables*self.k_N)
		#			, self.x_t_rec)
		self.EBTightCoupling = integrate.odeint(self.TightCouplingRegime2, np.reshape(self.BoltzmannTightCoupling, self.NumVarTightCoupling*self.k_N),
					self.x_t_rec, mxstep=10000)
		#print np.transpose(self.EBTightCoupling)
		print 'Tight coupling regime complete, now calculating after tight coupling'
		#print EBTightCoupling
		self.BoltzmannEinstein_InitConditions_AfterTC()
		EBAfterTC = integrate.odeint(self.BoltzmannEinstein_Equations, np.reshape(self.BoltzmannVariablesAFTERTC_INIT, self.NumVariables*self.k_N)\
				,self.x_t_today, mxstep = 20000)
		print 'Done, now plotting'
		print 'Time elapsed: ', (time.clock() - self.time_start)
		#print EBAfterTC
		EBSolutions = np.concatenate([self.BoltzmannVariablesAFTERTC, np.transpose(EBAfterTC)], axis=1)
		print 'HEYAYA'
		print EBSolutions
		print 'HEEE'
		print np.transpose(EBSolutions)
		print 'LENS'
		print len(EBSolutions)
		print len(np.transpose(EBSolutions))
		print len(np.transpose(EBSolutions)[0])

		print 'Writing to file'
		for ks in range(self.k_N):
			filename = "../VariableData/BoltzmannVariables_k" + str(ks) + ".txt"
			self.Write_Outfile(filename, EBSolutions, ks)
		

		#Transposed = np.transpose(EBSolutions)
		Transposed = EBSolutions
		#print EBTightCoupling
		#print Transposed[0]
		#print Transposed[1]
		plt.figure()
		plt.hold("on")
		plt.semilogy(self.x_t, Transposed[0])
		plt.semilogy(self.x_t, Transposed[1])
		#plt.semilogy(self.x_t_rec, Transposed[2])
		#plt.semilogy(self.x_t_rec, Transposed[3])
		#plt.semilogy(self.x_t_rec, Transposed[4])
		plt.legend(['k=0', 'k=1'])
		plt.xlabel('$x$')
		plt.ylabel('$\Theta_0$')
		plt.show()
		"""
		plt.figure()
		plt.hold("on")
		plt.plot(self.x_t_rec, Transposed[0])
		plt.plot(self.x_t_rec, Transposed[1])
		plt.plot(self.x_t_rec, Transposed[2])
		plt.plot(self.x_t_rec, Transposed[3])
		plt.plot(self.x_t_rec, Transposed[4])
		plt.legend(['k=0', 'k=1'])
		plt.xlabel('$x$')
		plt.ylabel('$Theta_0$')
		plt.show()
		"""
		"""
		if self.savefig == 1:
			a=1
		else:
			plt.show()
		"""
solver = time_mod(savefig=0, l_max=6)
solver.Plot_results(100)
#cProfile.run('solver.Plot_results(100)')

Array = np.array([[1,2],[10,20],[100,200]])
Array2 = np.array([[3,4,5],[30,40,50],[300,400,500]])
print np.concatenate([Array, Array2], axis=1)