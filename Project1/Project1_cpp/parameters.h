#ifndef PARAMETERS_H
#define PARAMETERS_H
#include <cmath>

class Parameters
{
public:
    Parameters();
    // Cosmological constants
    double Omega_b() const {return 0.046;}
    double Omega_m() const {return 0.224;}
    double Omega_r() const {return 8.3e-5;}
    double Omega_nu() const {return 0;}
    double Omega_lambda() const {return (1.0 - Omega_m() - Omega_b() - Omega_r() - Omega_nu());}
    double T_0() const {return 2.725;}
    double n_s() const {return 1.0;}
    double A_s() const {return 1.0;}
    double H_0() const {return (h0_*100*1e3)/Mpc_;}

    // General constants
    double c() const {return 2.99792458e8;}
    double epsilon_0() const {return 13.605698*eV_;}
    double m_e() const {return 9.10938188e-31;}
    double m_H() const {return 1.673534e-27;}
    double sigma_T() const {return 6.652462e-29;}
    double G_grav() const {return 6.67258e-11;}
    double rho_c() const {return (3*H_0()*H_0())/(8*pi_*G_grav());}
    double alpha() const {return 7.29735308e-3;}
    double hbar() const {return 1.05457148e-34;}
    double k_b() const {return 1.3806503e-23;}

private:
    // Units
    double pi_ = acos(-1);
    double h0_ = 0.7;
    double eV_ = 1.60217646e-19;
    double Mpc_ = 3.08568025e22;
};

#endif // PARAMETERS_H
