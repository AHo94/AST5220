#include <iostream> // Basic stuff
#include <fstream>  // Used to writeo outfiles
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
    double step = (end-start)/(double(NumberPts)-1);
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
    double HubbleParam = H_0*sqrt((Omega_m + Omega_b)*exp(-3.0*x0) + Omega_r*exp(-4.0*x0) + Omega_lambda);
    return HubbleParam;}

double Get_Hubble_prime(double x0){
    // Function that returns the scaled Hubble parameter a*H
    double Hprime = H_0*sqrt((Omega_m + Omega_b)*exp(-x0) + Omega_r*exp(-2.0*x0) + Omega_lambda*exp(2.0*x0));
    return Hprime;}

double Get_Hubble_prime_derivative(double x0){
    // Function that returns the derivative of the scaled Hubble parameter
    double HprimeDer = H_0Squared*(0.5*(Omega_m + Omega_b) + 2.0*Omega_r + 2.0*Omega_lambda)/Get_Hubble_prime(x0);
    return HprimeDer;}

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

int KroneckerDelta_2l(int l){
    // Kronecker Delta. Returns 1 if l=2.
    if (l == 2){return 1;}
    else{return 0;}
}

void Diff_eq_eta(const state_type &eta, state_type &detadt, double x0){
    // Right hand side of the diff. equation for conformal time
    detadt[0] = c/Get_Hubble_prime(x0);}

double Get_n_b(double x0){
    // Returns number density for baryons (or Hydrogen) for a given x
    double n_b = n_bConst/exp(3.0*x0);
    return n_b;}

double Saha_Equation(double x0){
    /* Solves saha equation. Only returns the positively valued solution of X_e.
       Assumes that the equation is in the form x^2 + bx + c = 0. Also assume c = -b.
       With a = 1, we can drop calculating the factor a and c.
    */
    double b = (Saha_b_factor/Get_n_b(x0))*exp(-EpsTemp_factor*exp(x0) - 3.0*x0/2.0);
    double SolPlus = (0.5)*(-b + sqrt(b*b + 4.0*b));
    double SolMinus = (0.5)*(-b - sqrt(b*b + 4.0*b));
    if (SolPlus > double(0.0)){
        return SolPlus;}
    else{
        return SolMinus;}
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
    double X_e_INIT = 1.0;
    double XeLimit = 0.965;
    ComputedX_e.push_back(X_e_INIT);
    int EndI;
    for (int i=0; i<n; i++){
        if (ComputedX_e[i] > XeLimit){
            double xe = Saha_Equation(x_values[i]);
            ComputedX_e.push_back(xe);}
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

    Diff_eq_tau(vector<double> n_eVec, vector<double> x_values) : m_Ne(n_eVec), m_x(x_values) {
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
    double v_b = c*k*Phi/(2.0*Hprime0);
    double Theta_0 = 0.5*Phi;
    double Theta_1 = -c*k*Phi/(6.0*Hprime0);
    InitialCond = {Theta_0, Theta_1, delta_b, delta_b, v_b, v_b, Phi};
}
struct Get_TC_end{
    /* Computes the time tight coupling ends.
    * Tight coupling is when k/(Hprimed*tau') << 1, i.e is very small
    * Assumes tight coupling end when k/(Hprimed*tau') > 0.1
    */
    vector<double> m_Taus;
    vector<double> m_x_values;
    spline1dinterpolant m_splineTCend;
    real_1d_array m_XInterp, m_TauInterp;

    Get_TC_end(vector<double> TauVec, vector<double> x_values) : m_Taus(TauVec), m_x_values(x_values) {
        m_XInterp.setcontent(m_x_values.size(), &(m_x_values[0]));
        m_TauInterp.setcontent(m_Taus.size(), &(m_Taus[0]));
        spline1dbuildcubic(m_XInterp, m_TauInterp, m_splineTCend);}

    double Get_xTCEnd(double k_value){
        vector<double> ckHTaus(m_Taus.size()+1);
        double s_temp, d2s_temp, TauDer;
        for (int i=0; i<m_Taus.size(); i++){
            spline1ddiff(m_splineTCend, m_x_values[i], s_temp, TauDer, d2s_temp);
            ckHTaus[i] = c*k_value/(Get_Hubble_prime(m_x_values[i])*TauDer);
        }
        for (int j=0; j<m_Taus.size(); j++){
            if (fabs(ckHTaus[j]) < double(0.1)){
                return m_x_values[j];
            }
        }
    }
};

struct Solve_TightCoupling{
    // Class that sets up interpolation for tight coupling regime calculation
    vector<double> m_TauTC;
    vector<double> m_xTC;
    double m_kTC;
    spline1dinterpolant m_splineTC;
    real_1d_array m_XInterp, m_TauInterp;
    Solve_TightCoupling(vector<double>TauVec, vector<double>x_values, double k_values)
        : m_TauTC(TauVec), m_xTC(x_values), m_kTC(k_values) {
        m_XInterp.setcontent(m_xTC.size(), &(m_xTC[0]));
        m_TauInterp.setcontent(m_TauTC.size(), &(m_TauTC[0]));
        spline1dbuildcubic(m_XInterp, m_TauInterp, m_splineTC);}
    void operator()(const state_type &Var, state_type &dVardx, double x0){
        // Solves Boltzmann equations for tight coupling
        double Theta_0 = Var[0];
        double Theta_1 = Var[1];
        double delta = Var[2];
        double delta_b = Var[3];
        double v = Var[4];
        double v_b = Var[5];
        double Phi = Var[6];

        double Hprimed = Get_Hubble_prime(x0);
        double HprimedDer = Get_Hubble_prime_derivative(x0);
        double Hprime_HPrimedDer = HprimedDer/Hprimed;
        double ck_Hprimed = (c*m_kTC)/Hprimed;
        double Tau, TauDer, TauDoubleDer;
        spline1ddiff(m_splineTC, x0, Tau, TauDer, TauDoubleDer);

        double R = 4.0*Omega_r/(3.0*Omega_b*exp(x0));
        double Theta_2 = -20.0*ck_Hprimed*Theta_1/(45.0*TauDer);
        double Psi = -Phi - PsiPrefactor*Omega_r*Theta_2/(m_kTC*m_kTC*exp(2.0*x0));
        double ck_HPsi = ck_Hprimed*Psi;
        dVardx[6] = Psi - ck_Hprimed*ck_Hprimed*Phi/3.0
            + (H_0Squared/(2.0*Hprimed*Hprimed))*(Omega_m*delta*exp(-x0) + Omega_b*delta_b*exp(-x0)
            + 4.0*Omega_r*exp(-2.0*x0)*Theta_0);
        dVardx[0] = -ck_Hprimed*Theta_1 - dVardx[6];
        double q = -(((1.0-2.0*R)*TauDer + (1.0+R)*TauDoubleDer)*(3.0*Theta_1 + v_b) - ck_HPsi
                + (1.0-Hprime_HPrimedDer)*ck_Hprimed*(-Theta_0 + 2.0*Theta_2) - ck_Hprimed*dVardx[0])
                /((1.0+R)*TauDer + Hprime_HPrimedDer - 1);
        dVardx[2] = ck_Hprimed*v - 3.0*dVardx[6];
        dVardx[3] = ck_Hprimed*v_b - 3.0*dVardx[6];
        dVardx[4] = -v - ck_HPsi;
        dVardx[5] = (-v_b - ck_HPsi + R*(q + ck_Hprimed*(-Theta_0 + 2.0*Theta_2) - ck_HPsi))/(1.0+R);
        dVardx[1] = (q-dVardx[5])/3.0;
        /*
        dVardx[0] = dTheta0dx;
        dVardx[1] = dTheta1dx;
        dVardx[2] = dDeltadx;
        dVardx[3] = dDelta_bdx;
        dVardx[4] = dvdx;
        dVardx[5] = dv_bdx;
        dVardx[6] = dPhidx;
        */
        /*
        cout << "----" << endl;
        cout << dVardx[0] << endl;
        cout << dVardx[1] << endl;
        cout << dVardx[2] << endl;
        cout << dVardx[3] << endl;
        cout << dVardx[4] << endl;
        cout << dVardx[5] << endl;
        cout << "dPhi: " << dVardx[6] << endl;
        cout << Var[6] << endl;
        cout << "aaa" << endl;
        cout << scientific;
        cout << setprecision(8) << endl;
        cout << "k = " << m_kTC << endl;
        cout << Psi << '\t' << c*m_kTC/(Hprimed) << '\t' << H_0 << '\t' << Hprimed << '\t' << Omega_m << '\t' << Var[2]
             << '\t' << exp(x0) << '\t' << Omega_b << '\t' << Var[3] << '\t' << Omega_r << '\t' << Var[0] << endl;
        exit(1);
        */
    }
};
struct Solve_BoltzmannEq{
    // Class that sets up interpolation for tight coupling regime calculation
    vector<double> m_Tau;
    vector<double> m_Eta;
    vector<double> m_xTau;
    vector<double> m_xEta;
    vector<double> m_l;
    double m_k;
    spline1dinterpolant m_TauSpline;
    spline1dinterpolant m_EtaSpline;
    real_1d_array m_XTauInterp, m_XEtaInterp, m_TauInterp, m_EtaInterp;
    Solve_BoltzmannEq(vector<double>TauVec, vector<double>x_Tau, vector<double> x_eta, vector<double> eta_values,
                      vector<double> l_Vector, double k_values)
        : m_Tau(TauVec), m_xTau(x_Tau), m_xEta(x_eta), m_Eta(eta_values), m_l(l_Vector), m_k(k_values) {
        m_XTauInterp.setcontent(m_xTau.size(), &(m_xTau[0]));
        m_XEtaInterp.setcontent(m_xEta.size(), &(m_xEta[0]));
        m_TauInterp.setcontent(m_Tau.size(), &(m_Tau[0]));
        m_EtaInterp.setcontent(m_Eta.size(), &(m_Eta[0]));
        spline1dbuildcubic(m_XTauInterp, m_TauInterp, m_TauSpline);
        spline1dbuildcubic(m_XEtaInterp, m_EtaInterp, m_EtaSpline);}
    void operator()(const state_type &Var, state_type &dVardx, double x0){
        // Solves Boltzmann equations for tight coupling
        double Theta_0 = Var[0];
        double Theta_1 = Var[1];
        double Theta_2 = Var[2];
        double Theta_3 = Var[3];
        double Theta_4 = Var[4];
        double Theta_5 = Var[5];
        double Theta_6 = Var[6];
        double delta = Var[7];
        double delta_b = Var[8];
        double v = Var[9];
        double v_b = Var[10];
        double Phi = Var[11];
        vector<double> Thetas(6);
        Thetas[0] = Theta_0, Thetas[1] = Theta_1, Thetas[2] = Theta_2, Thetas[3] = Theta_3;
        Thetas[4] = Theta_4, Thetas[5] = Theta_5, Thetas[6] = Theta_6;

        double Hprimed = Get_Hubble_prime(x0);
        double HprimedDer = Get_Hubble_prime_derivative(x0);
        double Hprime_HPrimedDer = HprimedDer/Hprimed;
        double ck_Hprimed = (c*m_k)/Hprimed;
        double Tau, TauDer, TauDoubleDer;
        spline1ddiff(m_TauSpline, x0, Tau, TauDer, TauDoubleDer);
        double EtaInterp = spline1dcalc(m_EtaSpline, x0);

        double R = 4.0*Omega_r/(3.0*Omega_b*exp(x0));
        double Psi = -Phi - PsiPrefactor*Omega_r*Theta_2/(m_k*m_k*exp(2.0*x0));
        double ck_HPsi = ck_Hprimed*Psi;
        dVardx[11] = Psi - ck_Hprimed*ck_Hprimed*Phi/3.0
            + (H_0Squared/(2.0*Hprimed*Hprimed))*(Omega_m*delta*exp(-x0) + Omega_b*delta_b*exp(-x0)
            + 4.0*Omega_r*exp(-2.0*x0)*Theta_0);

        dVardx[0] = -ck_Hprimed*Theta_1 - dVardx[11];
        dVardx[1] = ck_Hprimed*Theta_0/3.0 - 2.0*ck_Hprimed*Theta_2/3.0 + ck_HPsi/3.0 + TauDer*(Theta_1 + v_b/3.0);
        for (int l=2; l<6; l++){
            dVardx[l] = m_l[l]*ck_Hprimed*Thetas[l-1]/(2.0*m_l[l] + 1) - (m_l[l]+1)*ck_Hprimed/(2.0*m_l[l] + 1)
                    + TauDer*(Thetas[l] - 0.1*Thetas[l]*KroneckerDelta_2l(l));
        }
        dVardx[6] = ck_Hprimed*Theta_5 - 7.0*c*Theta_6/(Hprimed*EtaInterp) + TauDer*Theta_6;
        dVardx[7] = ck_Hprimed*v - 3.0*dVardx[11];
        dVardx[8] = ck_Hprimed*v_b - 3.0*dVardx[11];
        dVardx[9] = -v - ck_HPsi;
        /*
        cout << "dvar10 check " << endl;
        cout << v_b << endl;
        cout << ck_HPsi << endl;
        cout << TauDer << endl;
        cout << Theta_1 << endl;
        cout << EtaInterp << endl;
        cout << x0 << endl;
        cout << R << endl;
        */
        dVardx[10] = -v_b - ck_HPsi + TauDer*R*(3.0*Theta_1 + v_b);
    }
};

struct MergeAndFinalize{
    vector<double> m_Taus;
    vector<double> m_x;
    double m_k;
    spline1dinterpolant m_spline;
    real_1d_array m_XInterp, m_TauInterp;
    MergeAndFinalize(vector<double> TauVec, vector<double> x_values, double k_values)
        : m_Taus(TauVec), m_x(x_values), m_k(k_values) {
        m_XInterp.setcontent(m_x.size(), &(m_x[0]));
        m_TauInterp.setcontent(m_Taus.size(), &(m_Taus[0]));
        spline1dbuildcubic(m_XInterp, m_TauInterp, m_spline);}
    void MergeTC(vector<state_type> ComputedVar, vector<double> &Theta0, vector<double> &Theta1,
                 vector<double> &Theta2, vector<double> &Theta3, vector<double> &Theta4, vector<double> &Theta5,
                 vector<double> &Theta6, vector<double> &delta, vector<double> &deltab, vector<double> &v,
                 vector<double> &vb, vector<double> &Phi, vector<double> x_TC, state_type &InitCond){
        double TauHolder, TauDer, TauDDerHolder;
        int size = ComputedVar.size();
        for (int i=0; i<size-1; i++){
            double Hprime = Get_Hubble_prime(x_TC[i]);
            spline1ddiff(m_spline, x_TC[i], TauHolder, TauDer, TauDDerHolder);
            Theta0[i] = ComputedVar[i][0];
            Theta1[i] = ComputedVar[i][1];
            Theta2[i] = -20.0*c*m_k*Theta1[i]/(45.0*Hprime*TauDer);
            Theta3[i] = -3.0*c*m_k*Theta2[i]/(7.0*Hprime*TauDer);
            Theta4[i] = -4.0*c*m_k*Theta3[i]/(9.0*Hprime*TauDer);
            Theta5[i] = -5.0*c*m_k*Theta4[i]/(11.0*Hprime*TauDer);
            Theta6[i] = -6.0*c*m_k*Theta5[i]/(12.0*Hprime*TauDer);
            delta[i] = ComputedVar[i][2];
            deltab[i] = ComputedVar[i][3];
            v[i] = ComputedVar[i][4];
            vb[i] = ComputedVar[i][5];
            Phi[i] = ComputedVar[i][6];
        }
        cout << "Init cond" << endl;
        cout << "Theta0" << '\t' << Theta0[0] << endl;
        cout << "Theta1" << '\t' << Theta1[0] << endl;
        cout << "Theta2" << '\t' << Theta2[0] << endl;
        cout << "Theta3" << '\t' << Theta3[0] << endl;
        cout << "Theta4" << '\t' << Theta4[0] << endl;
        cout << "Theta5" << '\t' << Theta5[0] << endl;
        cout << "Theta6" << '\t' << Theta6[0] << endl;
        cout << "delta" << '\t' << delta[0] << endl;
        cout << "deltab" << '\t' << deltab[0] << endl;
        cout << "v" << '\t' << v[0] << endl;
        cout << "vb" << '\t' << vb[0] << endl;
        cout << "Phi" << '\t' << Phi[0] << endl;
        InitCond = {Theta0.back(), Theta1.back(), Theta2.back(), Theta3.back(), Theta4.back(), Theta5.back()
                    , Theta6.back(), delta.back(), deltab.back(), v.back(), vb.back(), Phi.back()};
    }
    void Finalize(vector<state_type> ComputedVar, vector<double> &Theta0, vector<double> &Theta1,
                  vector<double> &Theta2, vector<double> &Theta3, vector<double> &Theta4, vector<double> &Theta5,
                  vector<double> &Theta6, vector<double> &delta, vector<double> &deltab, vector<double> &v,
                  vector<double> &vb, vector<double> &Phi){
        for (int i=0; i<ComputedVar.size()-1; i++){
            Theta0.push_back(ComputedVar[i][0]);
            Theta1.push_back(ComputedVar[i][1]);
            Theta2.push_back(ComputedVar[i][2]);
            Theta3.push_back(ComputedVar[i][3]);
            Theta4.push_back(ComputedVar[i][4]);
            Theta5.push_back(ComputedVar[i][5]);
            Theta6.push_back(ComputedVar[i][6]);
            delta.push_back(ComputedVar[i][7]);
            deltab.push_back(ComputedVar[i][8]);
            v.push_back(ComputedVar[i][9]);
            vb.push_back(ComputedVar[i][10]);
            Phi.push_back(ComputedVar[i][11]);
        }
    }
};

void write_outfile(vector<double> x, vector<double> Value, string Value_name, string filename){
    // Saves on variable (as well as their corresponding x-values) to a text file.
    ofstream datafile;
    datafile.open(filename);
    datafile << "x" << setw(15) << Value_name << "\n";
    for (int i=0; i<Value.size(); i++){
        datafile << x[i] << setw(15) << Value[i] << '\n';
    }
    datafile.close();
}

void Write_Boltzmann_Variables(string filename, vector<double> x , vector<double> Theta0, vector<double> Theta1, vector<double> Theta2,
                          vector<double> Theta3, vector<double> Theta4, vector<double> Theta5, vector<double> Theta6,
                          vector<double> delta, vector<double> deltab, vector<double> v, vector<double> vb,
                          vector<double> Phi, double k_value){
    // Saves all boltzmann variables to one file
    ofstream datafile;
    datafile.open(filename);
    datafile << "x" << '\t' << "Theta0" << '\t' << "Theta1" << '\t' << "Theta2" << '\t' << "Theta3" << '\t'
             << "Theta4" << '\t' << "Theta5" << '\t' << "Theta6" << '\t' << "delta" << '\t'
             << "deltab" << '\t' << "v" << '\t'<< "vb" << '\t' << "Phi" << '\t' << "k=" << k_value << '\n';
    for (int i=0; i<x.size(); i++){
        datafile << x[i] << '\t' << Theta0[i] << '\t' << Theta1[i] << '\t' << Theta2[i] << '\t' << Theta3[i] << '\t'
                 << Theta4[i] << '\t' << Theta5[i] << '\t' << Theta6[i] << '\t' << delta[i] << '\t' << deltab[i] << '\t'
                 << v[i] << '\t' << vb[i] << '\t' << Phi[i] << '\n';
    }
    datafile.close();
}


void printstuf(const state_type &Var, double x0){
    cout << fixed;
    cout << setprecision(4) << x0 << '\t' << Var[0] << '\t' << Var[1] << '\t' << Var[2]
         << '\t' << Var[3] << '\t' << Var[4] << '\t' << Var[5] << '\t' << Var[6] << endl;
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

    vector<double> l_values(6);
    l_values[0] = 0.0; l_values[1] = 1.0; l_values[2] = 2.0; l_values[3] = 3.0;
    l_values[4] = 4.0; l_values[5] = 5.0; l_values[6] = 6.0;

    vector<double> k(100);
    double k_min = 0.1*H_0/c;
    double k_max = 1000*H_0/c;
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

    Get_TC_end ObtainXInstance(Taus, x_tau);

    double x_TC_end = ObtainXInstance.Get_xTCEnd(k[0]);

    vector<state_type> state_type_Temp;
    vector<double> x_TC;
    state_type Boltzmann_TC_Init(6);
    TightCoupling_InitialCondition(x_init, k[0], Boltzmann_TC_Init);
    Solve_TightCoupling TCInstance(Taus, x_tau, k[0]);
    timer = clock();

    size_t EBTC_step = integrate_adaptive(bulst_step(), TCInstance, Boltzmann_TC_Init, x_init, x_TC_end,
                       (x_TC_end-x_init)/(n_eta-1.0), // printstuf);
                       Save_single_variable(state_type_Temp, x_TC));

    for (int i=0; i<EBTC_step; i++){
        cout << state_type_Temp[i][0] << '\t' << state_type_Temp[i][1] << '\t'
             << state_type_Temp[i][2] << '\t' << state_type_Temp[i][3] << '\t'
             << state_type_Temp[i][4] << '\t' << state_type_Temp[i][5] << '\t'
             << state_type_Temp[i][6] << '\t' <<  endl;
    }

    MergeAndFinalize MergerInstance(Taus, x_tau, k[0]);
    vector<double> Theta0(EBTC_step);
    vector<double> Theta1(EBTC_step);
    vector<double> Theta2(EBTC_step);
    vector<double> Theta3(EBTC_step);
    vector<double> Theta4(EBTC_step);
    vector<double> Theta5(EBTC_step);
    vector<double> Theta6(EBTC_step);
    vector<double> delta(EBTC_step);
    vector<double> deltab(EBTC_step);
    vector<double> v(EBTC_step);
    vector<double> vb(EBTC_step);
    vector<double> Phi(EBTC_step);
    vector<state_type> States_Final;
    vector<double> x_final;
    state_type Init_afterTC(12);

    MergerInstance.MergeTC(state_type_Temp, Theta0, Theta1, Theta2, Theta3, Theta4, Theta5, Theta6,
                           delta, deltab, v, vb, Phi, x_TC, Init_afterTC);
    Solve_BoltzmannEq BoltzmannEQInstance(Taus, x_tau, x_etas, Etas, l_values, k[0]);
    integrate_adaptive(bulst_step(), BoltzmannEQInstance, Init_afterTC, x_TC_end, x_0,
                       (x_0 - x_TC_end)/(n_eta-1), Save_single_variable(States_Final, x_final));
    MergerInstance.Finalize(States_Final, Theta0, Theta1, Theta2, Theta3, Theta4, Theta5, Theta6,
                            delta, deltab, v, vb, Phi);
    vector<double> Full_x_grid;
    for (int i=0; i<x_TC.size(); i++){
        Full_x_grid.push_back(x_TC[i]);
    }
    for (int i=0; i<x_final.size(); i++){
        Full_x_grid.push_back(x_final[i]);
    }
    for (int i=0; i<Full_x_grid.size(); i++){
        cout << Full_x_grid[i] << endl;
    }
    Write_Boltzmann_Variables("TEST.txt", Full_x_grid, Theta0, Theta1, Theta2, Theta3, Theta4,
                              Theta5, Theta6, delta, deltab, v, vb, Phi, k[0]);
    timer = clock() - timer;
    cout << "tot time " << float(timer)/CLOCKS_PER_SEC << endl;
    cout << "Number of datapoints for tight coupling: " << EBTC_step << endl;
    cout << state_type_Temp.size() << endl;
    return 0;
}
