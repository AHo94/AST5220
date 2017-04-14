#include <iostream> // Basic stuff
#include <fstream>  // Used to writeo utfiles
#include <boost/array.hpp>  // Boost arrays
#include <boost/numeric/odeint.hpp> // Boost odeint
#include <boost/tuple/tuple.hpp>    // Boost tuple, used to return multiple values
#include <constants.h>      // Precalculated constants
#include "spline.h"         // Cubic spline (linear) interpolator

using namespace std;
using namespace boost::numeric::odeint;
using namespace constants;

typedef std::vector<double> variables;

struct push_back_state_and_time{
    std::vector<variables>& m_states;
    std::vector<double>& m_times;
    push_back_state_and_time(std::vector<variables> &states, std::vector<double>&times)
    :m_states(states), m_times(times) {}

    void operator()(const variables &x, double t)
    {
        m_states.push_back(x);
        m_times.push_back(t);
    }
};

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

double Get_Hubble_param(double x0){
    // Function that returns the Hubble parameter
    return constants::H_0*sqrt((constants::Omega_m + constants::Omega_b)*exp(-3.0*x0)
                               +constants::Omega_r*exp(-4.0*x0) + constants::Omega_lambda);
}

double Get_Hubble_prime(double x0){
    // Function that returns the scaled Hubble parameter a*H
    return constants::H_0*sqrt((constants::Omega_m + constants::Omega_b)*exp(-x0)
                               +constants::Omega_r*exp(-2.0*x0) + constants::Omega_lambda*exp(2.0*x0));
}

double Get_Hubble_prime_derivative(double x0){
    // Function that returns the derivative of the scaled Hubble parameter
    return constants::H_0Squared*(0.5*(constants::Omega_m + constants::Omega_b)
                                  +2.0*constants::Omega_r + 2.0*constants::Omega_lambda)/Get_Hubble_prime(x0);
}

boost::tuple<double, double, double, double> Get_omegas(double x0){
    // Computes the Omegas for a given 'time' x
    double H = Get_Hubble_param(x0);
    double rho_c = constants::rhoCrit_factor*H*H;
    double Omega_m_z = constants::rho_m0*exp(-3.0*x0)/rho_c;
    double Omega_b_z = constants::rho_b0*exp(-3.0*x0)/rho_c;
    double Omega_r_z = constants::rho_r0*exp(-4.0*x0)/rho_c;
    double Omega_lambda_z = constants::rho_lambda0/rho_c;
    return boost::make_tuple(Omega_m_z, Omega_b_z, Omega_r_z, Omega_lambda_z);
}

void Diff_eq_eta(const variables &eta, variables &detadt, double x0){
    detadt[0] = constants::c/Get_Hubble_prime(x0);
}

void write_outfile(vector<double> x, vector<variables> Value, string Value_name, string filename){
    int vec_size = Value.size();
    cout << vec_size << endl;
    ofstream datafile;
    datafile.open(filename);
    datafile << "x" << setw(15) << Value_name << "\n";
    for (int i=0; i<vec_size; i++){
        datafile << x[i] << setw(15) << Value[0][i] << '\n';
    }
    datafile.close();
}

int main(int argc, char *argv[])
{
    int n1 = 200;
    int n2 = 300;
    int n_t = n1+n2;

    double z_start_rec = 1630.4;
    double z_end_rec = 614.2;
    double z_0 = 0.0;
    double x_start_rec = -log(1.0 + z_start_rec);
    double x_end_rec = -log(1.0 + z_end_rec);
    double x_0 = 0.0;

    int n_eta = 3000;
    double a_init = 1e-11;
    double x_eta_init = log(a_init);
    double x_eta_end = 0.0;
    vector<double> x_eta(n_eta), x_tau(n_eta);
    linspace(x_eta_init, x_eta_end, n_eta, x_eta);

    vector<variables> Etas;
    vector<double> x_etas;
    variables eta_init = {0};
    //double eta_init = 0;
    typedef runge_kutta4<double> RK4_stepper;
    //integrate_adaptive(RK4_stepper(), Diff_eq_eta, eta_init, x_eta_init,
    //                   x_eta_end, (x_eta_end - x_eta_init)/(n_eta-1),
    //                   Save_single_variable(Etas));
    size_t steps = integrate(Diff_eq_eta, eta_init, x_eta_init,
                       x_eta_end, (x_eta_end - x_eta_init)/(n_eta-1),
                       Save_single_variable(Etas, x_etas));
    cout << "steps" << steps << endl;
    for (int i=0; i<steps; i++){
        cout << x_etas[i] << '\t' << Etas[i][0] << endl;
    }
    //write_outfile(x_eta, Etas, "Eta", "Eta_test.txt");

    /*
    double Omm, Omb, Omr, Oml;
    boost::tuple<double, double, double, double> Omegas = Get_omegas(0.0);
    cout << Omegas.get<0>() << '\t' << Omegas.get<1>() << '\t' << Omegas.get<2>() << '\t' << Omegas.get<3>() << endl;

    std::vector<double> X(5), Y(5);
    X[0]=0.1; X[1]=0.4; X[2]=1.2; X[3]=1.8; X[4]=2.0;
    Y[0]=0.1; Y[1]=0.7; Y[2]=0.6; Y[3]=1.1; Y[4]=0.9;
    tk::spline s;
    s.set_points(X,Y);    // currently it is required that X is already sorted
    double x=1.5;
    printf("spline at %f is %f\n", x, s(x));

    vector<double> a1(11), a2(11);
    linspace(0, 10.0, 11, a1);
    linspace(2, 12, 11, a2);
    */
    return 0;
}
