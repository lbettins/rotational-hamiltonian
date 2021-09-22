#define _USE_MATH_DEFINES

#include <fstream>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <complex>
#include <cstdio>
#include <chrono>
#include <string>
#include <armadillo>
#include <vector>
#include "wignerSymbols.h"
//#include "wigner/gaunt.hpp"

//using namespace std::chrono;
using namespace arma;

static const double kB = 1.3806504e-23;         // J/K
static const double NA = 6.02214179e23;         // 1/mol
static const double EHARTREE = 4.35974434e-18;  // J/Hartree
static const double AMU = 1.660538921e-27;      // kg/amu
static const double HBAR = 1.054571726e-34;     // J.s
static const double HBAR1 = HBAR / EHARTREE;    // Hartree.s
static const double HBAR2 = HBAR * 1e20 / AMU;  // amu.Å^2/s
static const double SCH4 = 186.25; // J/mol.K

Mat<complex<double>> getHamiltonian(int lmax, double Ix, double Iy, double Iz, std::vector<double>& a);
SpMat<complex<double>> getSparseHam(int lmax, double Ix, double Iy, double Iz, std::vector<double>& a);
//Col<double> getCoefficients(std::string sysname);
std::vector<double> getWignerCoeffs(std::string sysname, bool freeRotor);
std::vector<double> getMomentOfInertia(std::string sysname="METH-CHA");
std::string getDirectory(std::string sysname);
double getPartitionFunction(double T, Mat<complex<double>>& H, int sym=1);
double getSparseQ(double T, SpMat<complex<double>>& H, int sym=1);

int main(int argc, char** argv) {
    /* argv[0]  Program name
     * argv[1]  System name
     * argv[2]  Lmax (start)
     * argv[3]  free-rotor
     * argv[4]  temperature / K
     */
    if (argc != 5) {
        throw std::runtime_error("Enter systemName as it appears in data directory, Lmax, free-rotor flag, temperature in Kelvin");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::string sysname = argv[1];
    bool freeRotor = std::stoi(argv[3]);
    double T = std::stod(argv[4]);
    std::string dirname = getDirectory(sysname);
    std::cout << "Directory containing data is: " << dirname << std::endl;
    std::vector<double> a = getWignerCoeffs(sysname, freeRotor);
    int sigma = 1;

    std::vector<double> Ivec = getMomentOfInertia(sysname);
    std::cout << "Moments of Inertia:" << std::endl;
    for (double i : Ivec) {
        std::cout << i << '\t';
    }
 
    /*
     *  Rotational Constants + Classical Partition Function
     */
    double B = HBAR1*HBAR2/(2.0*Ivec[2]);
    double A = HBAR1*HBAR2/(2.0*Ivec[1]);
    double C = HBAR1*HBAR2/(2.0*Ivec[0]);
    //std::cout << "kB T / Hartree:\n" << kB * 298 / EHARTREE << std::endl;
    std::cout << "Qapprox = " << sqrt(M_PI)/sigma * sqrt(pow(kB*300/EHARTREE, 3) / (A*B*C)) << std::endl;  

    /*
     *  Dense Matrix Implementation 
     */
    bool converge = false;
    bool dense = true;
    int lmax = atoi(argv[2]);
    double Q;
    double Qprev = 0;
    std::cout << "Lmax\t\tQ\t\tTime\t\tdQ" << std::endl;
    do {
        auto qstart = std::chrono::high_resolution_clock::now();
        if (dense) {
            /****  Dense Matrix Implementation  ****/
            unsigned long long size = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3.0;
            Mat<complex<double>> H = getHamiltonian(lmax, Ivec[0], Ivec[1], Ivec[2], a);
            if (!H.is_hermitian(1e-3)) {
                H.brief_print("H = ");
            }
            try {
                Q = getPartitionFunction(298.15, H, sigma);
            } catch (const std::logic_error& e) {
                //std::cout << "Required Memory is " << size*size*8*1e-9 << " Gb." << std::endl; 
                //std::cout << "Memory problem caused failure. Switching to sparse implementation." << std::endl;
                //dense = false;
                throw e;
                break;
            }
            /* *********************************** */
        } else {
            break;
            /****  Sparse Matrix Implementation  ****/
            SpMat<complex<double>> H = getSparseHam(lmax, Ivec[0], Ivec[1], Ivec[2], a);
            Q = getSparseQ(298.15, H, sigma);
            /****************************************/
        }
        auto qend = std::chrono::high_resolution_clock::now();
        auto qduration = std::chrono::duration_cast<std::chrono::microseconds>(qend - qstart);
        double dQ = fabs(Q-Qprev);
        std::cout << lmax << "\t\t" << Q << "\t\t" << qduration.count()/1e6 << " sec." << "\t\t" << dQ << std::endl;
        if (dQ < 1e-4) {
            converge = true;
            std::cout << "DeltaQ = " << fabs(Q-Qprev) << std::endl;
            std::cout << "Convergence criterion met!" << std::endl;
            std::cout << "Lmax = " << lmax << std::endl;
        }
        lmax++;
        Qprev = Q;
    } while (!converge);
    std::cout << std::endl;
    std::cout << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << std::endl << duration.count()/1e6 << " secs" << std::endl;
    return 0;
}


double getPartitionFunction(double T, Mat<complex<double>>& H, int sym) {
    /* Solve the Eigenvalues
     * Inputs:  T;  the temperature [=] K
     *          H;  the Hamiltonian matrix [=] Hartree
     */
    double b = pow(kB * T, -1) * EHARTREE;
    double tr = 0;

    //std::cout << "Solving Matrix exponential" << std::endl;
    //Mat<complex<double>> expbH = expmat_sym(-b*H);
    vec eigval = eig_sym(H);
    for (double e : eigval) {
        tr += exp(-b*e);
    }
    //std::cout  <<  "Solved Matrix exponential" << std::endl;
    //double tr = trace(expmat_sym(-b*H));
    //std::cout << std::endl << "Q predicted by eig_sym: " << tr/sym << std::endl;
    return double(tr/sym);
    //return double(Q/sym);
}

Mat<complex<double>> getHamiltonian(int lmax, double Ix, double Iy, double Iz, std::vector<double>& a) {
    /* Construct the Hamiltonian Matrix
     * Inputs:  lmax;   the maximum quantum number forthe spherical basis
     *             I;   the gas-phase moments of inertia [=] amu*Å^2
     *             a;   coefficients for potential in the Wigner D Matrix basis
     * Outputs:    H;   the Hamiltonian matrix (Hermitian) [=] Hartree 
     */
    long double size = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3.0;
    Mat<complex<double>> H = zeros<mat>(size, size);

    // Define rotational constants:
    double B = HBAR1*HBAR2/(2.0*Iz);
    double A = HBAR1*HBAR2/(2.0*Iy);
    double C = HBAR1*HBAR2/(2.0*Ix);
    double kap = (A == B && A == C) ? 0 : (2.0*B - (A+C)) / (A-C);
    #pragma omp parallel
    {
        unsigned long long i = 0;
        #pragma omp for
        for (int el = 0; el < lmax+1; el++) {
            for (int m = -el; m <= el; m++) {
                for (int k = -el; k <= el; k++) {
                    //unsigned long long i = (4*el*el*el/3.0) + 2*el*el + (5*el/3.0) + 2*m*el + m + k;
                    //std::cout << j << '\t';
                    unsigned long long j = 0;
                    for (int ell = 0; ell < lmax+1; ell++) {
                        for (int mm = -ell; mm <= ell; mm++) {
                            for (int kk = -ell; kk <= ell; kk++) {
                                //unsigned long long j = (4*ell*ell*ell/3.0) + 2*ell*ell + (5*ell/3.0) + 2*mm*ell + mm + kk;
                                if (j > i) continue;
                                if (i == j) {
                                    try {
                                        H(i,j) += 0.5*(A+C)*el*(el+1) + 0.5*(A-C)*kap*k*k;
                                        if (k+2 <= el) {
                                            double val = 0.25*(C-A)*sqrt(el*(el+1)-k*(k+1))*sqrt(el*(el+1)-(k+1)*(k+2));
                                            H(i+2,j) += val;
                                            H(j,i+2) += val;
                                        }
                                    } catch (const std::exception& e) {
                                        std::cout << "Failure at index: " <<
                                            i << "\t(" << el << ',' << m << ',' <<
                                            k << ')' << std::endl;
                                    }
                                }
                                //if (a.size() == 1) continue;
                                //double Vij = 0;
                                //for (int L = 0; L < 8; L++) {
                                //    for (int M = -L; M <= L; M++) {
                                //        for (int K = -L; K <= L; K++) {
                                //            unsigned long long ind = (4*L*L*L/3.0) + 2*L*L + (5*L/3.0) + 2*M*L + M + K;
                                //            if (ind > a.size()-1 || a[ind] == 0) continue;
                                //            //double Clm = WignerSymbols::clebschGordan(L,ell,el,M,mm,m);
                                //            //double Clk = WignerSymbols::clebschGordan(L,ell,el,K,kk,k);
                                //            //double val = 8*M_PI*M_PI*Clm*Clk/(2.0*el+1.0);
                                //            double Wlm = WignerSymbols::wigner3j(L,ell,el,M,mm,-m);
                                //            double Wlk = WignerSymbols::wigner3j(L,ell,el,K,kk,-k);
                                //            double val = 8*M_PI*M_PI*pow(-1.0, -m-k)*Wlm*Wlk;
                                //            Vij += a[ind] * val;
                                //        }
                                //    }
                                //}
                                //H(i,j) += Vij;
                                //H(j,i) += Vij;
                                j++;
                            }
                        }
                    }
                    i++;
                }
            }
        }
    } // end parallel
    //H.print("H = ");
    //std::cout << "Constructed matrix with LMAX " << lmax << '.' << std::endl;
    return H;
}

std::vector<double> getWignerCoeffs(std::string sysname, bool freeRotor) {
    if (freeRotor) {
        std::vector<double> a = { 0.0 };
        return a;
    }
    std::string dirname = getDirectory(sysname);
    std::string filename = dirname+"/a.txt";
    std::ifstream is(filename);
    double val;
    std::vector<double> a;
    while (is) {
        if(!(is >> val)) {
            break;
        }
        a.push_back(val);
    }
    return a;
}

std::vector<double> getMomentOfInertia(std::string sysname) {
    std::string dirname = getDirectory(sysname);
    std::string filename = dirname+'/'+"I.txt";
    std::ifstream is(filename);
    double val;
    std::vector<double> Ivec;
    while (is) {
        if (!(is >> val)) {
            break;
        }
        Ivec.push_back(val);
    }
    return Ivec;
}

std::string getDirectory(std::string sysname) {
    //std::string dirname = "/Users/lancebettinson/Thesis/umrr/code/hamiltonian-cpp/data";
    std::string dirname = "/global/scratch/lbettins/rotational-hamiltonian/data";
    if (sysname == "") {
        std::string sysname;
        std::cout << "Enter system name:" << std::endl;
        std::cin >> sysname;
    }
    return dirname+'/'+sysname;
}

SpMat<complex<double>> getSparseHam(int lmax, double Ix, double Iy, double Iz, std::vector<double>& a) {
    /* Construct the Hamiltonian Matrix
     * Inputs:  lmax;   the maximum quantum number forthe spherical basis
     *             I;   the gas-phase moments of inertia [=] amu*Å^2
     * Outputs:    H;   the Hamiltonian matrix (Hermitian) [=] Hartree 
     */
    unsigned long long size = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3.0;
    SpMat<complex<double>> H = sp_mat(size, size);

    // Define rotational constants:
    double B = HBAR1*HBAR2/(2.0*Iz);
    double A = HBAR1*HBAR2/(2.0*Iy);
    double C = HBAR1*HBAR2/(2.0*Ix);

    double kap = (A == B && A == C) ? 0 : (2.0*B - (A+C)) / (A-C);
    #pragma omp parallel
    {
        #pragma omp for
        for (int el = 0; el < lmax+1; el++) {
            for (int m = -el; m <= el; m++) {
                for (int k = -el; k <= el; k++) {
                    unsigned long long i = (4*el*el*el/3.0) + 2*el*el + (5*el/3.0) + 2*m*el + m + k;
                    for (int ell = 0; ell < lmax+1; ell++) {
                        for (int mm = -ell; mm <= ell; mm++) {
                            for (int kk = -ell; kk <= ell; kk++) {
                                unsigned long long j = (4*ell*ell*ell/3.0) + 2*ell*ell + (5*ell/3.0)
                                    + 2*mm*ell + mm + kk;
                                if (j > i) continue;
                                if (i == j) {
                                    try {
                                        H(i,j) += 0.5*(A+C)*el*(el+1) + 0.5*(A-C)*kap*k*k;
                                        if (k+2 <= el) {
                                            double val = 0.25*(C-A)*sqrt(el*(el+1)-k*(k+1))*sqrt(el*(el+1)-(k+1)*(k+2));
                                            H(i+2,j) += val;
                                            H(j,i+2) += val;
                                        }
                                    } catch (const std::exception& e) {
                                        std::cout << "Failure at index: " <<
                                            i << "\t(" << el << ',' << m << ',' <<
                                            k << ')' << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // end parallel
    return H;
}

double getSparseQ(double T, SpMat<complex<double>>& H, int sym) {
    /* Solve the Eigenvalues forSparse Matrix
     * Inputs:  T;  the temperature [=] K
     *          H;  the (sparse) Hamiltonian matrix [=] Hartree
     */
    double b = pow(kB * T, -1) * EHARTREE;
    vec eigval;
    mat eigvec;
    //std::cout << "Number of rows: " << H.n_rows << std::endl;
    eigs_sym(eigval, eigvec, H, H.n_rows-1);
    double Q = 0;
    //std::cout << "Eigenvalues: " << std::endl;
    for (double e : eigval) {
        //std::cout << e << '\t';
        Q += exp(-b * e);
    }
    //std::cout << "Q predicted by eigs_sym: " << Q/sym << std::endl;
    return double(Q/sym);
}

Col<double> getCoefficients(std::string sysname) {
    std::string dirname = getDirectory(sysname);
    std::string filename = dirname+'/'+"vdat.txt";
    std::ifstream is(filename);
    if (is.fail())
    {
        std::cout << "cannot open file " << filename;
    }
    double theta, phi, v;
    std::vector<double> my_vec;
    while (is) {
        if (!(is >> theta >> phi >> v)) {
            break;
        }
        //std::cout << theta << '\t' << phi << '\t' << v << std::endl;
        my_vec.push_back(v);
    }
    Col<double> cvec = conv_to<vec>::from(my_vec);
    //cvec.print();
    is.close();
    return cvec;
}
