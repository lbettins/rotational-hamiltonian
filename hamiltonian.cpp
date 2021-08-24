#define _USE_MATH_DEFINES

#include <fstream>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>
#include <armadillo>
#include <vector>
#include "wigner/gaunt.hpp"

using namespace std::chrono;
using namespace arma;

static const double kB = 1.3806504e-23;         // J/K
static const double NA = 6.02214179e23;         // 1/mol
static const double EHARTREE = 4.35974434e-18;  // J/Hartree
static const double AMU = 1.660538921e-27;      // kg/amu
static const double HBAR = 1.054571726e-34;     // J.s
static const double HBAR1 = HBAR / EHARTREE;    // Hartree.s
static const double HBAR2 = HBAR * 1e20 / AMU;  // amu.Å^2/s
static const double SCH4 = 186.25; // J/mol.K

Mat<double> getHamiltonian(int lmax, double Ix, double Iy, double Iz) {
    /* Construct the Hamiltonian Matrix
     * Inputs:  lmax;   the maximum quantum number for the spherical basis
     *             I;   the gas-phase moments of inertia [=] amu*Å^2
     * Outputs:    H;   the Hamiltonian matrix (Hermitian) [=] Hartree 
     */
    std::cout << "Lmax = " << lmax << std::endl;
    double size = (double(1.0/3.0)*(lmax+1)*(2*lmax+1)*(2*lmax+3));
    std::cout << "Dimensions of the matrix: " << size << std::endl;
    Mat<double> H = zeros<mat>(size+1, size+1);
    //SpMat<double> H = sp_mat(size, size);

    // Define rotational constants:
    double B = HBAR1*HBAR2/(2.0*Iz);
    double A = HBAR1*HBAR2/(2.0*Iy);
    double C = HBAR1*HBAR2/(2.0*Ix);

    double kap;
    if (A == B && A == C) {     //SPHERICAL ROTOR
        kap = 0;
    } else {
        kap = (2.0*B - (A + C)) / (A - C);
    }

    std::cout << "kappa val is " << kap << "." << std::endl;
    #pragma omp parallel
    {
        double g;
        double a;
        #pragma omp for
        for (int el = 0; el < lmax+1; el++) {
            for (int m = -el; m <= el; m++) {
                for (int k = -el; k <= el; k++) {
                    unsigned long long j = (4*el*el*el/3.0) + 2*el*el + (5*el/3.0) + 2*m*el + m + k;
                    //std::cout << j << '\t';
                    for (int ell = 0; ell < lmax+1; ell++) {
                        for (int mm = -ell; mm <= ell; mm++) {
                            for (int kk = -ell; kk <= ell; kk++) {
                                unsigned long long i = (4*ell*ell*ell/3.0) + 2*ell*ell + (5*ell/3.0)
                                    + 2*mm*ell + mm + kk;
                                //if (j < i) {
                                //    continue;
                                //} else {
                                //    H(i,j) += 1;
                                //}
                                if (i == j) {
                                    try {
                                        //H(i,j) += B*el*(el+1);
                                        H(i,j) += 0.5*(A+C)*el*(el+1) + 0.5*(A-C)*kap*k*k;
                                        if (k-2 >= -el) {
                                            H(i-2,j) += 0.25*(C-A)*sqrt(el*(el+1)-k*(k-1))*sqrt(el*(el+1)-(k-1)*(k-2));
                                            if (abs(double(j - i+2)) != 2) {
                                                std::cout << "(i-2, j) index spacing incorrect @ (" << i-2 << ',' << j << ") --> (" 
                                                    << el << ',' << m << ',' << k << ')' << std::endl;
                                            }

                                        }
                                        if (k+2 <= el) {
                                            H(i+2,j) += 0.25*(C-A)*sqrt(el*(el+1)-k*(k+1))*sqrt(el*(el+1)-(k+1)*(k+2));
                                            if (abs(double(i+2 - j)) != 2) {
                                                std::cout << "(i+2, j) index spacing incorrect @ (" << i+2 << ',' << j << ") --> (" 
                                                    << el << ',' << m << ',' << k << ')' << std::endl;
                                            }

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
    std::cout << std::endl;
    //H.print("H = ");
    return H;
}

SpMat<double> getSparseHam(int lmax, double Ix, double Iy, double Iz) {
    /* Construct the Hamiltonian Matrix
     * Inputs:  lmax;   the maximum quantum number for the spherical basis
     *             I;   the gas-phase moments of inertia [=] amu*Å^2
     * Outputs:    H;   the Hamiltonian matrix (Hermitian) [=] Hartree 
     */
    std::cout << "Lmax = " << lmax << std::endl;
    double size = (double(1.0/3.0)*(lmax+1)*(2*lmax+1)*(2*lmax+3));
    std::cout << "Dimensions of the sparse matrix: " << size << std::endl;
    SpMat<double> H = sp_mat(size, size);

    // Define rotational constants:
    double B = HBAR1*HBAR2/(2.0*Iz);
    double A = HBAR1*HBAR2/(2.0*Iy);
    double C = HBAR1*HBAR2/(2.0*Ix);

    double kap;
    if (A == B && A == C) {     //SPHERICAL ROTOR
        kap = 0;
    } else {
        kap = (2.0*B - (A + C)) / (A - C);
    }
    #pragma omp parallel
    {
        double g;
        double a;
        #pragma omp for
        for (int el = 0; el < lmax+1; el++) {
            for (int m = -el; m <= el; m++) {
                for (int k = -el; k <= el; k++) {
                    unsigned long long j = (4*el*el*el/3.0) + 2*el*el + (5*el/3.0) + 2*m*el + m + k;
                    //std::cout << j << '\t';
                    for (int ell = 0; ell < lmax+1; ell++) {
                        for (int mm = -ell; mm <= ell; mm++) {
                            for (int kk = -ell; kk <= ell; kk++) {
                                unsigned long long i = (4*ell*ell*ell/3.0) + 2*ell*ell + (5*ell/3.0)
                                    + 2*mm*ell + mm + kk;
                                //if (j < i) {
                                //    continue;
                                //} else {
                                //    H(i,j) += 1;
                                //}
                                if (i == j) {
                                    try {
                                        //H(i,j) += B*el*(el+1);
                                        H(i,j) += 0.5*(A+C)*el*(el+1) + 0.5*(A-C)*kap*k*k;
                                        if (k-2 >= -el) {
                                            H(i-2,j) += 0.25*(C-A)*sqrt(el*(el+1)-k*(k-1))*sqrt(el*(el+1)-(k-1)*(k-2));
                                            if (isnan(H(i-2,j))) {
                                                std:: cout << "NaN @ (" << i-2 << ',' << j << ") --> (" << el << ',' << m << ',' << k << ')' << std::endl;
                                            }
                                            if (abs(double(j - i+2)) != 2) {
                                                std::cout << "Index spacing incorrect @ (" << i-2 << ',' << j << ") --> (" << el << ',' << m << ',' << k << ')' << std::endl;
                                            }
                                        }
                                        if (k+2 <= el) {
                                            H(i+2,j) += 0.25*(C-A)*sqrt(el*(el+1)-k*(k+1))*sqrt(el*(el+1)-(k+1)*(k+2));
                                            if (isnan(H(i+2,j))) {
                                                std:: cout << "NaN @ (" << i+2 << ',' << j << ") --> (" << el << ',' << m << ',' << k << ')' << std::endl;
                                            }
                                            if (abs(double(i+2 - j)) != 2) {
                                                std::cout << "Index spacing incorrect @ (" << i+2 << ',' << j << ") --> (" << el << ',' << m << ',' << k << ')' << std::endl;
                                            }
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


std::string getDirectory(std::string sysname) {
    std::string dirname = "/Users/lancebettinson/Thesis/umrr/code/hamiltonian-cpp/data";
    if (sysname == "") {
        std::string sysname;
        std::cout << "Enter system name:" << std::endl;
        std::cin >> sysname;
    }
    return dirname+'/'+sysname;
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

std::vector<double> getMomentOfInertia(std::string sysname="METH-CHA") {
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

double getPartitionFunction(double T, Mat<double>& H, int sym=1) {
    /* Solve the Eigenvalues
     * Inputs:  T;  the temperature [=] K
     *          H;  the Hamiltonian matrix [=] Hartree
     */
    double b = pow(kB * T, -1) * EHARTREE;

    //Col<double> eigval = eig_sym(H);
    //double Q = 0;
    ////b = 1;
    //int count = 0;
    //for (double e : eigval) {
    //    Q += exp(-b * e);
    //}
    //for (double e : eigval) {
    //    if (count == 5) {
    //        //std::cout << std::endl;
    //    } //std::cout << exp(-b*e)/Q << '\t';
    //    count++;
    //}
    //std::cout << std::endl << "Q predicted by eig_sym: " << Q/sym << std::endl;
    Mat<double> bH = -b*H;
    Mat<double> expbH = expmat_sym(bH);
    double tr = trace(expbH);
    std::cout << pow(b,-1) << std::endl;
    std::cout << std::endl << "Q predicted by eig_sym: " << tr/sym << std::endl;
    return double(tr/sym);
    //return double(Q/sym);
}

double getSparseQ(double T, SpMat<double>& H, int sym=1) {
    /* Solve the Eigenvalues for Sparse Matrix
     * Inputs:  T;  the temperature [=] K
     *          H;  the (sparse) Hamiltonian matrix [=] Hartree
     */
    double b = pow(kB * T, -1) * EHARTREE;
    //SpMat<double> bH = -b*H;

    vec eigval;
    mat eigvec;
    
    std::cout << "Number of rows: " << H.n_rows << std::endl;
    eigs_sym(eigval, eigvec, H, H.n_rows-1);
    double Q = 0;
    std::cout << "Eigenvalues: " << std::endl;
    for (double e : eigval) {
        std::cout << e << '\t';
        Q += exp(-b * e);
    }
    std::cout << std::endl;
    std::cout << "Q predicted by eigs_sym: " << Q/sym << std::endl;
    return double(Q/sym);
}

int main() {
    auto start = high_resolution_clock::now();

    std::string sysname = "ETH1-CHA";
    int sigma = 1;
    std::string dirname = getDirectory(sysname);

    std::vector<double> Ivec = getMomentOfInertia(sysname);
    std::cout << "Printing I:" << std::endl;
    for (double i : Ivec) {
        std::cout << i << '\t';
    }
    //std::cout << "Rotational Constant: = " << HBAR1*HBAR2/2.0/I << std::endl;
    std::cout << "kT = " << kB*300 / EHARTREE << std::endl;
    //std::cout << "Ratio = " << HBAR1*HBAR2/2.0/I *EHARTREE/(kB*300) << std::endl;
    //std::cout << "Qapprox = " << kB*300/EHARTREE / (HBAR1*HBAR2/2.0/I) << std::endl;

    Col<double> ahat = getCoefficients(sysname);
    std::cout << "Directory is: " << dirname << std::endl;

    /*
     *  Sparse Matrix Implementation
     */
    //SpMat<double> spH = getSparseHam(25, Ivec[0], Ivec[1], Ivec[2]);
    ////spH.print("My sparse Ham =");
    //if (!spH.is_hermitian()) {
    //    std::cout << "NOT HERMITIAN, CHECK" << std::endl;
    //} else {
    //    std::cout << "IS HERMITIAN, GOOD TO GO" << std::endl;
    //}
    //double spQ = getSparseQ(300, spH, 3);
    //std::cout << "spQ = " << spQ << std::endl;

    
    /*
     *  Dense Matrix Implementation 
     */
    Mat<double> H = getHamiltonian(31, Ivec[0], Ivec[1], Ivec[2]);
    if (!H.is_hermitian()) {
        std::cout << "NOT HERMITIAN, CHECK" << std::endl;
    } else {
        std::cout << "HERMITIAN YAY" << std::endl;
    }
    //H.print("My ham:");
    double Q = getPartitionFunction(298, H, sigma);

    /*
     *  Classical Partition Function
     */
    double B = HBAR1*HBAR2/(2.0*Ivec[2]);
    double A = HBAR1*HBAR2/(2.0*Ivec[1]);
    double C = HBAR1*HBAR2/(2.0*Ivec[0]);
    std::cout << "Rotational constants / Hartree:\n" << A << '\t' << B << '\t' << C << '\t' << std::endl;
    std::cout << "kB T / Hartree:\n" << kB * 298 / EHARTREE << std::endl;
    std::cout << "Qapprox = " << sqrt(M_PI)/sigma * sqrt(pow(kB*300/EHARTREE, 3) / (A*B*C));

    std::cout << std::endl;
    std::cout << std::endl;
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << std::endl << duration.count()/1e6 << " secs" << std::endl;
    return 0;
}
