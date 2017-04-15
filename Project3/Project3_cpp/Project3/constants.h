#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <math.h>
using namespace std;

namespace  constants
{
    // Units
    double pi = acos(-1);
    const double eV = 1.60217647e-19;
    const double Mpc = 3.08568025e22;

    // Cosmological parameters
    const double Omega_b = 0.046;
    const double Omega_m = 0.224;
    const double Omega_r = 8.3e-5;
    const double Omega_nu = 0.0;
    const double Omega_lambda = 1.0 - Omega_m - Omega_b - Omega_r - Omega_nu;
    const double T_0 = 2.725;
    const double n_s = 1.0;
    const double A_s = 1.0;
    const double h0 = 0.7;
    const double H_0 = h0*100.0*1e3/Mpc;
    const double H_0Squared = H_0*H_0;

    // General constants
    const double c = 2.99792458e8;
    const double epsilon_0 = 13.605698*eV;
    const double m_e = 9.10938188e-31;
    const double m_H = 1.673534e-27;
    const double sigma_T = 6.652462e-29;
    const double G_grav = 6.67258e-11;
    const double rho_c0 = (3.0*H_0Squared)/(8*pi*G_grav);
    const double alpha = 7.29735308e-3;
    const double hbar = 1.05457148e-34;
    const double k_b = 1.3806503e-23;

    // Density Parameters today
    const double rho_m0 = Omega_m*rho_c0;
    const double rho_b0 = Omega_b*rho_c0;
    const double rho_r0 = Omega_r*rho_c0;
    const double rho_lambda0 = Omega_lambda*rho_c0;

    // Precalculate certain factors to reduce number of float point operations
    const double Saha_b_factor = pow((m_e*T_0*k_b)/(2.0*pi*hbar*hbar), 3.0/2.0); // Factor in front of 'b' in Saha equation
    const double rhoCrit_factor = 3.0/(8*pi*G_grav);                  // Used for critical density at arbitrary times

    // Constant used for Peebles equation and some constant factors that can be precalculated
    const double Lambda_2sto1s = 8.227;
    const double alpha_factor = ((64.0*pi)/(sqrt(27.0*pi)))*(alpha*alpha/(m_e*m_e))*(hbar*hbar/c);
    const double beta_factor = pow((m_e*T_0*k_b)/(2.0*pi), 3.0/2.0)*(1.0/(hbar*hbar*hbar));
    const double Lambda_alpha_factor = pow(3.0*epsilon_0/(hbar*c), 3.0)/(64.0*pi*pi);
    const double EpsTemp_factor = epsilon_0/(k_b*T_0);

    // Other constants
    const double n_bConst = Omega_b*rho_c0/m_H;
}
#endif // CONSTANTS_H
