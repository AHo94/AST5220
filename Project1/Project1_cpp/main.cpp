#include <iostream>
#include "parameters.h"
using namespace std;

int main(int argc, char *argv[])
{
    Parameters params;

    double n1 = 200;    // # Grid points during recombination
    double n2 = 300;    // # Grid points after recombination
    double n_t = n1 + n2;   // Total # grid points
    double z_start_rec = 1630.4;    // Redshift at the start of recombination
    double z_end_rec = 614.2;       // Redshift at the end of recombination
    double z_0 = 0;                 // Redshift today
    double x_start_rec = -log(1.0 + z_start_rec);
    double x_end_rec = -log(1.0 + z_end_rec);
    double x_0 = 0;

    double n_eta = 1000;
    double a_init = 1.0*pow(10,-10);
    double x_eta1 = log(a_init);
    double x_eta2 = 0;

    // Creating new grid fills
    double *x_grid = new double[n_t];
    double *a_grid = new double[n_t];
    return 0;
}
