#include <iostream> // Basic stuff
#include <fstream>  // Used to writeo utfiles
#include <boost/array.hpp>  // Boost arrays
#include <boost/numeric/odeint.hpp> // Boost odeint
#include <boost/tuple/tuple.hpp>    // Boost tuple, used to return multiple values
#include <constants.h>      // Precalculated constants
#include <math.h>     // Mathematical expressions.
#include "interpolation.h"  // Alglib interpolation
#include <time.h>

using namespace std;
using namespace boost::numeric::odeint;
using namespace constants;
using namespace alglib;

typedef std::vector<double> state_type;
typedef runge_kutta4<state_type> rk4_step;
typedef bulirsch_stoer<state_type> bulst_step;
typedef bulirsch_stoer_dense_out<state_type> bulst_step_dense;
typedef runge_kutta_cash_karp54<state_type> rk_fehl;

struct Save_single_variable{
    // Saves the computed values from a differential equation to a vector
    vector<state_type> &m_states;
    vector<double> &m_x;
    Save_single_variable(vector<state_type>&var_state, vector<double>&x):m_states(var_state),m_x(x) {}
    void operator()(const state_type &value, double x0)
    {
        m_states.push_back(value);
        m_x.push_back(x0);
    }
};
/*
struct Save_state_type{
    // Saves the computed state_type from the Boltzmann equations
    std::vector<state_type> &m_states;
    std::vector<double> &m_x;
    Save_single_variable(std::vector<state_type>&var_state, std::vector<double>&x):m_states(var_state),m_x(x) {}
    void operator()(const state_type &value, double x0)
    {
        m_states.push_back(value);
        m_x.push_back(x0);
    }
};
*/
void linspace(double start, double end, int NumberPts, vector<double> &array){
    // Creating a function equivalent to linspace from Numpy in Python
    double step = (end-start)/(NumberPts-1);
    for (int i=0; i<NumberPts; i++){
        array[i] = start + i*step;}
}

void Sort_variable_to_vector_SingleVar(vector<state_type> VariableArr, vector<double>&OutputArr, int Arrsize, int merge=0){
    /* If merge = 0 (default), converts type vector<state_type> to vector<double>.
     * If merge = 1, merges two vectors together. Specifically used for the electron number density vector.
    */
    if (merge == 0){
        for (int i=0; i<=Arrsize; i++){
            OutputArr[i] = VariableArr[i][0];}
    }
    else if (merge == 1){
        for (int i=0; i<=Arrsize; i++){
            OutputArr.push_back(VariableArr[i][0]);}
    }
    else{
        cout << "Argument merge not properly set. Try merge=0 or merge=1" << endl;
        exit(EXIT_FAILURE);
    }
}

double Get_Hubble_param(double x0){
    // Function that returns the Hubble parameter
    return H_0*sqrt((Omega_m + Omega_b)*exp(-3.0*x0) + Omega_r*exp(-4.0*x0) + Omega_lambda);}

double Get_Hubble_prime(double x0){
    // Function that returns the scaled Hubble parameter a*H
    return H_0*sqrt((Omega_m + Omega_b)*exp(-x0) + Omega_r*exp(-2.0*x0) + Omega_lambda*exp(2.0*x0));}

double Get_Hubble_prime_derivative(double x0){
    // Function that returns the derivative of the scaled Hubble parameter
    return H_0Squared*(0.5*(Omega_m + Omega_b) + 2.0*Omega_r + 2.0*Omega_lambda)/Get_Hubble_prime(x0);}

boost::tuple<double, double, double, double> Get_omegas(double x0){
    // Computes the Omegas for a given 'time' x
    double H = Get_Hubble_param(x0);
    double rho_c = rhoCrit_factor*H*H;
    double Omega_m_z = rho_m0*exp(-3.0*x0)/rho_c;
    double Omega_b_z = rho_b0*exp(-3.0*x0)/rho_c;
    double Omega_r_z = rho_r0*exp(-4.0*x0)/rho_c;
    double Omega_lambda_z = rho_lambda0/rho_c;
    return boost::make_tuple(Omega_m_z, Omega_b_z, Omega_r_z, Omega_lambda_z);
}

void Diff_eq_eta(const state_type &eta, state_type &detadt, double x0){
    // Right hand side of the diff. equation for conformal time
    detadt[0] = c/Get_Hubble_prime(x0);}

double Get_n_b(double x0){
    // Returns number density for baryons (or Hydrogen) for a given x
    return n_bConst/exp(3.0*x0);}

double Saha_Equation(double x0){
    /* Solves saha equation. Only returns the positively valued solution of X_e.
       Assumes that the equation is in the form x^2 + bx + c = 0. Also assume c = -b.
       With a = 1, we can drop calculating the factor a and c.
    */
    double b = (Saha_b_factor/Get_n_b(x0))*exp(-EpsTemp_factor*exp(x0) - 3.0*x0/2.0);
    if (0.5*(-b + sqrt(b*b + 4.0*b)) > 0){
        return 0.5*(-b + sqrt(b*b + 4.0*b));}
    else if (0.5*(-b - sqrt(b*b + 4.0*b)) > 0){
        return 0.5*(-b - sqrt(b*b + 4.0*b));}
}
void Peebles_equation(const state_type &X_e, state_type &dXedx, double x0){
    // Returns the right hand side of Peebles equation
    double n_b = Get_n_b(x0);
    double H = Get_Hubble_param(x0);
    double exp_factor = EpsTemp_factor*exp(x0);
    double phi2 = 0.448*log(exp_factor);
    double alpha2 = alpha_factor*sqrt(exp_factor)*phi2;
    double beta = alpha2*beta_factor*exp(-3.0*x0/2.0 - exp_factor);
    double beta2 = alpha2*beta_factor*exp(-3.0*x0/2.0 - exp_factor/4.0);
    double Lambda_alpha = H*Lambda_alpha_factor/((1.0-X_e[0])*n_b);
    double C_r = (Lambda_2sto1s + Lambda_alpha)/(Lambda_2sto1s + Lambda_alpha + beta2);
    dXedx[0] = (C_r/H)*(beta*(1.0 - X_e[0]) - n_b*alpha2*X_e[0]*X_e[0]);
}

void Compute_Xe(int n, double x_init, double x_0, vector<double> &ComputedX_e){
    // Computes X_e
    vector<double> x_n_e;
    vector<double> x_values(n);
    vector<state_type> X_e_temp;
    linspace(x_init, x_0, n, x_values);
    ComputedX_e.push_back(1.0);
    int EndI;
    for (int i=0; i<n; i++){
        if (ComputedX_e[i] > 0.97){
            ComputedX_e.push_back(Saha_Equation(x_values[i]));}
        else{
            EndI = i;
            break;}
    }
    state_type X_e_init = {ComputedX_e[EndI]};
    size_t X_e_size = integrate_adaptive(rk4_step(), Peebles_equation, X_e_init, x_values[EndI],
                            x_0, (x_0 - x_values[EndI])/(n-EndI-1), Save_single_variable(X_e_temp, x_n_e));
    Sort_variable_to_vector_SingleVar(X_e_temp, ComputedX_e, X_e_size-1, 1);
}

struct Diff_eq_tau{
    /* A class for the optical depth differential equation.
     * Sets up interpolation for given set of x and log(n_e) values.
     */
    vector<double>& m_Ne;
    vector<double>& m_x;
    spline1dinterpolant m_spline;
    real_1d_array m_XInterp, m_NeInterp;

    Diff_eq_tau(vector<double>&n_eVec, vector<double>&x_values) : m_Ne(n_eVec), m_x(x_values) {
        m_XInterp.setcontent(m_x.size(), &(m_x[0]));
        m_NeInterp.setcontent(m_Ne.size(), &(m_Ne[0]));
        spline1dbuildcubic(m_XInterp, m_NeInterp, m_spline);}

    void operator()(const state_type &tau, state_type &dtaudx, double x0){
        // Calculates the right hand side of the optical depth diff.eq.
        dtaudx[0] = -exp(spline1dcalc(m_spline, x0))*sigma_T*c/Get_Hubble_param(x0);
    }
};

void TightCoupling_InitialCondition(double x0,  double k, state_type &InitialCond){
    /* Sets the initial conditions for the relevant state_type in the tight coupling regime.
     * Array order is Theta0, Theta1, delta, delta_b, v, v_b, Phi.
     */
    double Hprime0 = Get_Hubble_prime(x0);
    double Phi = 1.0;
    double delta_b = 3.0*Phi/2.0;
    double v_b = k*Phi/(2.0*Hprime0);
    double Theta_0 = 0.5*Phi;
    double Theta_1 = -c*k*Phi/(6.0*Hprime0);
    InitialCond = {Theta_0, Theta_1, delta_b, delta_b, v_b, v_b, Phi};
}

struct Solve_TightCoupling{
    // Class that sets up interpolation for tight coupling regime calculation
    vector<double> m_TauTC;
    vector<double> m_xTC;
    double m_kTC;
    spline1dinterpolant m_splineTC;
    real_1d_array m_XInterp, m_TauInterp;
    Solve_TightCoupling(vector<double>&TauVec, vector<double>&x_values, double &k_values)
        : m_TauTC(TauVec), m_xTC(x_values), m_kTC(k_values) {
        m_XInterp.setcontent(m_xTC.size(), &(m_xTC[0]));
        m_TauInterp.setcontent(m_TauTC.size(), &(m_TauTC[0]));
        spline1dbuildcubic(m_XInterp, m_TauInterp, m_splineTC);}
    void operator()(const state_type &Var, state_type &dVardx, double x0){
        // Solves Boltzmann equations for tight coupling
        double Om_m, Om_b, Om_r, Om_lambda;
        boost::tuple<double, double, double, double> Omegas = Get_omegas(x0);
        Om_m = Omegas.get<0>();
        Om_b = Omegas.get<1>();
        Om_r = Omegas.get<2>();
        Om_lambda = Omegas.get<3>();
        double Hprimed = Get_Hubble_prime(x0);
        double HprimedDer = Get_Hubble_prime_derivative(x0);
        double Hprime_HPrimedDer = Hprimed/HprimedDer;
        double ck_Hprimed = c*m_kTC/Hprimed;
        double Tau, TauDer, TauDoubleDer;
        spline1ddiff(m_splineTC, x0, Tau, TauDer, TauDoubleDer);

        double R = 4.0*Om_r/(3.0*Om_m*exp(x0));
        double Theta2 = -20.0*ck_Hprimed*Var[1]/(45.0*TauDer);

        double Psi = -Var[6] - PsiPrefactor*Om_r*Theta2/(m_kTC*m_kTC*exp(2.0*x0));
        double ck_HPsi = ck_Hprimed*Psi;
        dVardx[6] = - Psi - ck_Hprimed*ck_Hprimed*Var[6]/3.0
            + (H_0Squared/(2.0*Hprimed*Hprimed))*(Om_m*Var[2]*exp(-x0) + Om_b*Var[3]*exp(-x0) + 4*Om_r*exp(-x0)*Var[0]);
        dVardx[0] = -ck_Hprimed*Var[1] - dVardx[6];
        double q = -(((1.0-2.0*R)*TauDer + (1.0+R)*TauDoubleDer)*(3.0*Var[1] + Var[5]) - ck_HPsi
                + (1.0-Hprime_HPrimedDer)*ck_Hprimed*(-Var[0] + 2*Theta2) - ck_Hprimed*dVardx[0])
                /((1.0+R)*TauDer + Hprime_HPrimedDer - 1);
        dVardx[2] = ck_Hprimed*Var[4] - 3.0*dVardx[6];
        dVardx[3] = ck_Hprimed*Var[5] - 3.0*dVardx[6];
        dVardx[4] = -Var[4] - ck_HPsi;
        dVardx[5] = (-Var[5] - ck_HPsi + R*(q + ck_Hprimed*(-Var[0] + 2*Theta2) - ck_HPsi))/(1.0+R);
        dVardx[1] = (q-dVardx[5])/3.0;

    }
};

void write_outfile(vector<double> x, vector<double> Value, string Value_name, string filename){
    // Saves data to text file.
    int vec_size = Value.size();
    ofstream datafile;
    datafile.open(filename);
    datafile << "x" << setw(15) << Value_name << "\n";
    for (int i=0; i<vec_size; i++){
        datafile << x[i] << setw(15) << Value[i] << '\n';
    }
    datafile.close();
}
/*
void Sort_Boltzmann_variables(vector<state_type> Input_vector, vector<double> &Theta0, vector<double> &Thet1,
      vector<double> &Theta2, vector<double> &Theta3, vector<double> &Theta4, vector<double> &Theta5,
      vector<double> &Theta6, vector<double> &delta, vector<double> &delta_b, vector<double> &v,
      vector<double> &v_b, vector<double> &phi, int TightCoupling = 1){
    // Sorting the boltzmann variables to their seperate
    if (TightCoupling == 1){

    }
}
*/
void printstuf(const state_type &Var, double x0){
    cout << fixed;
    cout << setprecision(10) << x0 << '\t' << Var[0] << endl;
}


int main(int argc, char *argv[])
{

    // Initializing some arrays
    clock_t timer;
    int n1 = 200;
    int n2 = 300;
    int n_t = n1+n2;

    double z_start_rec = 1630.4;
    double z_end_rec = 614.2;
    double x_start_rec = -log(1.0 + z_start_rec);
    double x_end_rec = -log(1.0 + z_end_rec);
    double x_0 = 0.0;

    int n_eta = 3000;
    double a_init = 1e-8;
    double x_init = log(a_init);

    vector<double> x_TC(200);
    vector<double> x_after_TC(300);
    linspace(x_init, x_start_rec, n1, x_TC);
    linspace(x_start_rec, x_end_rec, n2, x_after_TC);

    vector<double> k(100);
    double k_min = 0.1*H_0;
    double k_max = 10*H_0;
    for (int i=0; i<100; i++){
        k[i] = k_min + (k_max-k_min)*(i*i/10000.0);
    }

    // Solving for conformal time
    vector<state_type> Etas_temp;
    vector<double> x_etas;
    state_type eta_init = {0};
    size_t eta_size = integrate_adaptive(rk4_step(), Diff_eq_eta, eta_init, x_init,
                  x_0, (x_0 - x_init)/(n_eta-1.0), Save_single_variable(Etas_temp, x_etas));
    vector<double> Etas(eta_size+1);
    Sort_variable_to_vector_SingleVar(Etas_temp, Etas, eta_size); // Sorting Etas_temp to Etas

    // Computing X_e
    vector<double> X_e;
    vector<double> x_eta2(n_eta);
    linspace(x_init, x_0, n_eta, x_eta2);
    Compute_Xe(n_eta, x_init, x_0, X_e);

    // Stores n_e (logarithmic scale) to an array. Used to interpolate for taus
    vector<double> LOGn_e(X_e.size());
    for (int i=0; i<X_e.size(); i++){
        LOGn_e[i] = log(X_e[i]*Get_n_b(x_eta2[i]));}

    // Calculate tau
    vector<state_type> Taus_temp;
    vector<double> Taus(n_eta);
    vector<double> x_tau;
    state_type tau_init = {0};
    Diff_eq_tau Taudif_instance(LOGn_e, x_eta2);
    size_t Tau_size = integrate_adaptive(rk4_step(), Taudif_instance, tau_init,
                    x_0, x_init, (x_init-x_0)/(n_eta-1), Save_single_variable(Taus_temp, x_tau));
    Sort_variable_to_vector_SingleVar(Taus_temp, Taus, Tau_size);

    // Interpolate derivatives of tau
    int n_doublederPtS = 100;
    vector<double> TauDerivative(n_eta);
    vector<double> TauDoubleDer(n_doublederPtS);
    vector<double> x_TauDoubleDer(n_doublederPtS);
    linspace(x_0, x_init, n_doublederPtS, x_TauDoubleDer);
    real_1d_array TAU_Interp, X_Interp;
    spline1dinterpolant splineTaus;
    TAU_Interp.setcontent(Taus.size(), &(Taus[0]));
    X_Interp.setcontent(x_tau.size(), &(x_tau[0]));
    spline1dbuildcubic(X_Interp, TAU_Interp, splineTaus);
    double s_temp, d2s_temp;
    for (int i=0; i<n_eta; i++){
        spline1ddiff(splineTaus, x_tau[i], s_temp, TauDerivative[i], d2s_temp);}
    for (int j=0; j<n_doublederPtS; j++){
        spline1ddiff(splineTaus, x_TauDoubleDer[j], s_temp, d2s_temp, TauDoubleDer[j]);}

    // Compute visibility function and its derivatives (interpolated)
    vector<double> g(n_eta);
    vector<double> gDer(n_eta);
    vector<double> gDoubleDer(n_eta);
    for (int i=0; i<n_eta; i++){
        g[i] = -TauDerivative[i]*exp(-Taus[i]);}
    spline1dinterpolant splineGs;
    real_1d_array G_Interp, X_Interp2;
    G_Interp.setcontent(g.size(), &(g[0]));
    spline1dbuildcubic(X_Interp, G_Interp, splineGs);
    for (int i=0; i<n_eta; i++){
        spline1ddiff(splineGs, x_tau[i], s_temp, gDer[i], gDoubleDer[i]);}

    vector<state_type> state_type_Temp;
    vector<double> x_var;
    state_type state_type_InitTC(6);
    TightCoupling_InitialCondition(x_init, k[0], state_type_InitTC);
    Solve_TightCoupling TCInstance(Taus, x_tau, k[0]);
    cout << "Instance ok" << endl;
    timer = clock();
    size_t EBTC_step = integrate_adaptive(bulst_step(), TCInstance, state_type_InitTC, x_init, x_start_rec,
                       (x_start_rec-x_init)/(n_eta-1.0),
                       Save_single_variable(state_type_Temp, x_var));
    timer = clock() - timer;
    cout << "tot time " << float(timer)/CLOCKS_PER_SEC << endl;
    cout << "Number of datapoints for tight coupling: " << EBTC_step << endl;
    return 0;
}
