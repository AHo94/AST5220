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

typedef std::vector<double> variables;
typedef runge_kutta4<variables> rk4_step;
typedef bulirsch_stoer<variables> bulst_step;
typedef bulirsch_stoer_dense_out<variables> bulst_step_dense;

struct Save_single_variable{
    // Saves the computed values from a differential equation to a vector
    std::vector<variables> &m_states;
    std::vector<double> &m_x;
    Save_single_variable(std::vector<variables>&var_state, std::vector<double>&x):m_states(var_state),m_x(x) {}
    void operator()(const variables &value, double x0)
    {
        m_states.push_back(value);
        m_x.push_back(x0);
    }
};

void linspace(double start, double end, int NumberPts, vector<double> &array){
    // Creating a function equivalent to linspace from Numpy in Python
    double step = (end-start)/(NumberPts-1);
    for (int i=0; i<NumberPts; i++){
        array[i] = start + i*step;
    }
}

void Sort_variable_to_vector_SingleVar(vector<variables> VariableArr, vector<double>&OutputArr, int Arrsize, int merge=0){
    /* If merge = 0 (default), converts type vector<variables> to vector<double>.
     * If merge = 1, merges two vectors together. Specifically used for the electron number density vector.
    */
    if (merge == 0){
        for (int i=0; i<=Arrsize; i++){
            OutputArr[i] = VariableArr[i][0];
        }
    }
    else if (merge == 1){
        for (int i=0; i<=Arrsize; i++){
            OutputArr.push_back(VariableArr[i][0]);
        }
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

void Diff_eq_eta(const variables &eta, variables &detadt, double x0){
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
        return 0.5*(-b + sqrt(b*b + 4.0*b));
    }
    else if (0.5*(-b - sqrt(b*b + 4.0*b)) > 0){
        return 0.5*(-b - sqrt(b*b + 4.0*b));
    }
}
void Peebles_equation(const variables &X_e, variables &dXedx, double x0){
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
    vector<variables> X_e_temp;
    linspace(x_init, x_0, n, x_values);
    ComputedX_e.push_back(1.0);
    int EndI;
    for (int i=0; i<n; i++){
        if (ComputedX_e[i] > 0.97){
            ComputedX_e.push_back(Saha_Equation(x_values[i]));
        }
        else{
            EndI = i;
            break;
        }
    }
    variables X_e_init = {ComputedX_e[EndI]};
    size_t X_e_size = integrate_adaptive(rk4_step(), Peebles_equation, X_e_init, x_values[EndI],
                                         x_0, (x_0 - x_values[EndI])/(n-EndI-1),
                                         Save_single_variable(X_e_temp, x_n_e));
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

    void operator()(const variables &tau, variables &dtaudx, double x0){
        // Calculates the right hand side of the optical depth diff.eq.
        dtaudx[0] = -exp(spline1dcalc(m_spline, x0))*sigma_T*c/Get_Hubble_param(x0);
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
    double x_eta_init = log(a_init);

    // Solving for conformal time
    vector<variables> Etas_temp;
    vector<double> x_etas;
    variables eta_init = {0};
    size_t eta_size = integrate_adaptive(rk4_step(), Diff_eq_eta, eta_init, x_eta_init,
                  x_0, (x_0 - x_eta_init)/(n_eta-1.0),
                  Save_single_variable(Etas_temp, x_etas));

    vector<double> Etas(eta_size+1);
    Sort_variable_to_vector_SingleVar(Etas_temp, Etas, eta_size); // Sorting Etas_temp to Etas

    // Computing X_e
    vector<double> X_e;
    vector<double> x_eta2(n_eta);
    linspace(x_eta_init, x_0, n_eta, x_eta2);
    Compute_Xe(n_eta, x_eta_init, x_0, X_e);
    write_outfile(x_eta2, X_e, "X_e", "X_e_test.txt");

    // Stores n_e (logarithmic scale) to an array. Used to interpolate for taus
    vector<double> LOGn_e(X_e.size());
    for (int i=0; i<X_e.size(); i++){
        LOGn_e[i] = log(X_e[i]*Get_n_b(x_eta2[i]));
    }
    // Calculate tau
    vector<variables> Taus_temp;
    vector<double> Taus(n_eta);
    vector<double> x_tau;
    variables tau_init = {0};
    Diff_eq_tau Taudif_instance(LOGn_e, x_eta2);
    size_t Tau_size = integrate_adaptive(rk4_step(), Taudif_instance, tau_init,
                                         x_0, x_eta_init, (x_eta_init-x_0)/(n_eta-1),
                                         Save_single_variable(Taus_temp, x_tau));

    Sort_variable_to_vector_SingleVar(Taus_temp, Taus, Tau_size);
    write_outfile(x_tau, Taus, "Tau", "TauTest.txt");

    // Interpolate derivatives of tau
    int n_doublederPtS = 100;
    vector<double> TauDerivative(n_eta);
    vector<double> TauDoubleDer(n_doublederPtS);
    vector<double> x_TauDoubleDer(n_doublederPtS);
    linspace(x_0, x_eta_init, n_doublederPtS, x_TauDoubleDer);
    real_1d_array TAU_Interp, X_Interp;
    spline1dinterpolant splineTaus;
    TAU_Interp.setcontent(Taus.size(), &(Taus[0]));
    X_Interp.setcontent(x_tau.size(), &(x_tau[0]));
    spline1dbuildcubic(X_Interp, TAU_Interp, splineTaus);
    double s_temp, d2s_temp;
    for (int i=0; i<n_eta; i++){
        spline1ddiff(splineTaus, x_tau[i], s_temp, TauDerivative[i], d2s_temp);
    }
    for (int j=0; j<n_doublederPtS; j++){
        spline1ddiff(splineTaus, x_TauDoubleDer[j], s_temp, d2s_temp, TauDoubleDer[j]);
    }
    write_outfile(x_tau, TauDerivative, "TauDer", "TauDerTest.txt");
    write_outfile(x_TauDoubleDer, TauDoubleDer, "TauDoubleDer", "TauDoublDerText.txt");

    // Compute visibility function and its derivatives (interpolated)
    vector<double> g(n_eta);
    vector<double> gDer(n_eta);
    vector<double> gDoubleDer(n_eta);
    for (int i=0; i<n_eta; i++){
        g[i] = -TauDerivative[i]*exp(-Taus[i]);
    }
    spline1dinterpolant splineGs;
    real_1d_array G_Interp, X_Interp2;
    G_Interp.setcontent(g.size(), &(g[0]));
    //X_Interp2.setcontent(x_tau.size(), &(x_tau[0]));
    spline1dbuildcubic(X_Interp, G_Interp, splineGs);
    for (int i=0; i<n_eta; i++){
        spline1ddiff(splineGs, x_tau[i], s_temp, gDer[i], gDoubleDer[i]);
    }
    write_outfile(x_tau, g, "g", "gTest.txt");
    write_outfile(x_tau, gDer, "gDer", "gDerTest.txt");
    write_outfile(x_tau, gDoubleDer, "gDoubleDer", "gDoubleDerTest.txt");

    /*
    cout << eta_size << endl;
    vector<double> Etas(eta_size+1);
    for (int i=0; i<=eta_size; i++){
        Etas[i] = Etas_temp[i][0];
    }

    write_outfile(x_etas, Etas, "Eta", "Eta_test.txt");
    vector<double> interp_x_eta(100);
    cout << "Linspacing" << endl;
    linspace(x_start_rec, x_end_rec, 100, interp_x_eta);
    real_1d_array Xs, etay;
    cout << "set content 1" << endl;
    Xs.setcontent(x_etas.size(), &(x_etas[0]));
    cout << x_etas.size() << '\t' << Etas.size() << endl;
    cout << "set content 2" << endl;
    etay.setcontent(Etas.size(), &(Etas[0]));
    cout << "Interp setup" << endl;
    spline1dinterpolant spline;
    cout << "Interp setup 2" << endl;
    spline1dbuildcubic(Xs, etay, spline);
    cout << "Interp eta define" << endl;
    vector<double> InterpEta(100);
    cout << "Doing interp" << endl;
    for (int i=0; i<100; i++){
        InterpEta[i] = spline1dcalc(spline,interp_x_eta[i]);
    }
    write_outfile(interp_x_eta, InterpEta, "EtaInterpol", "InterpTest.txt");
    */
    /*
    double Omm, Omb, Omr, Oml;
    boost::tuple<double, double, double, double> Omegas = Get_omegas(0.0);
    cout << Omegas.get<0>() << '\t' << Omegas.get<1>() << '\t' << Omegas.get<2>() << '\t' << Omegas.get<3>() << endl;

    std::vector<double> X(5), Y(5);
    X[0]=0.1; X[1]=0.4; X[2]=1.2; X[3]=1.8; X[4]=2.0;
    Y[0]=0.1; Y[1]=0.7; Y[2]=0.6; Y[3]=1.1; Y[4]=0.9;
    alglib::real_1d_array AX, AY;
    AX.setcontent(X.size(), &(X[0]));
    AY.setcontent(Y.size(), &(Y[0]));
    alglib::spline1dinterpolant spline;
    alglib::spline1dbuildcubic(AX, AY, X.size(), 2,0.0,2,0.0,spline);

    for(size_t i=0; i<X.size(); i++){
       printf("%f %f\n", X[i], Y[i]);
    }
    printf("\n");
    for(int i=-50; i<250; i++){
       double x=0.01*i;
       double s, ds, d2s;
       alglib::spline1ddiff(spline,x,s,ds,d2s);
       printf("%f %f %f %f\n", x, s, ds, d2s);
    }
    printf("\n");
    */
    /*
    tk::spline s;
    s.set_points(X,Y);    // currently it is required that X is already sorted
    for (int i =-50; i<250; i++){
        double x=0.01*i;
        printf("%f %f %f \n", x, s(x), s.deriv(1,x), s.deriv(2,x));
    }
    */

    return 0;
}
