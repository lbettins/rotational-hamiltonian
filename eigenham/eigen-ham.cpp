#define EIGEN_USE_MKL
#define EIGEN_DONT_PARALLELIZE
#define _USE_MATH_DEFINES

#include <fstream>
#include <iostream>
#include <omp.h>    // make sure intel module is loaded
#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <vector>

//using namespace std::chrono;
using namespace Eigen;

static const double kB = 1.3806504e-23;         // J/K
static const double NA = 6.02214179e23;         // 1/mol
static const double EHARTREE = 4.35974434e-18;  // J/Hartree
static const double AMU = 1.660538921e-27;      // kg/amu
static const double HBAR = 1.054571726e-34;     // J.s
static const double HBAR1 = HBAR / EHARTREE;    // Hartree.s
static const double HBAR2 = HBAR * 1e20 / AMU;  // amu.Å^2/s
static const double SCH4 = 186.25; // J/mol.K

MatrixXd getHamiltonian(int lmax, double Ix, double Iy, double Iz);
std::vector<double> getMomentOfInertia(std::string sysname="METH-CHA");
std::string getDirectory(std::string sysname);
double getPartitionFunction(double T, MatrixXd& H, int sym=1);

int main(int argc, char** argv) {
    /* argv[0]  Program name
     * argv[1]  System name
     * argv[2]  Lmax
     */
    if (argc != 4) {
        throw std::runtime_error("Enter systemName as it appears in data directory, Lmax, & n_omp_threads");
    }
    initParallel();
    omp_set_num_threads(atoi(argv[3]));
    //setNbThreads(32);
    std::cout << "Multithread run using " << argv[3] << " threads." << std::endl;
    std::cout << "Starting clock." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::string sysname = argv[1];
    int sigma = 1;
    std::string dirname = getDirectory(sysname);
    std::cout << "Extracting data from directory " << dirname << std::endl;
    std::vector<double> Ivec = getMomentOfInertia(sysname);
    std::cout << "Moments of Inertia: ";
    for (double i : Ivec) {
        std::cout << i << '\t';
    }
    std::cout << std::endl;

    //Col<double> ahat = getCoefficients(sysname);
    
    bool converge = false;
    double Qprev = 0;
    double Q;
    int lmax = atoi(argv[2]);
    std::cout << "Lmax\t\tQ\t\tTime" << std::endl;
    //std::cout << "Constructing Hamiltonian and calculating partition function." << std::endl;
    do {
        auto qstart = std::chrono::high_resolution_clock::now();
        /****  Dense Matrix Implementation  ****/
        MatrixXd H = getHamiltonian(lmax, Ivec[0], Ivec[1], Ivec[2]);
        Q = getPartitionFunction(298, H, sigma);
        /* *********************************** */
        auto qend = std::chrono::high_resolution_clock::now();
        auto qduration = std::chrono::duration_cast<std::chrono::microseconds>(qend - qstart);
        std::cout << lmax << "\t\t" << Q << "\t\t" << qduration.count()/1e6 << " sec." << std::endl;
        if (abs(Q-Qprev) < 1e-6) {
            converge = true;
            std::cout << "DeltaQ = " << abs(Q-Qprev) << std::endl;
            std::cout << "Convergence criterion met!" << std::endl;
        }
        lmax++;
        Qprev = Q;
    } while (!converge);

    /****  Classical Partition Function ****/
    double B = HBAR1*HBAR2/(2.0*Ivec[2]);
    double A = HBAR1*HBAR2/(2.0*Ivec[1]);
    double C = HBAR1*HBAR2/(2.0*Ivec[0]);
    std::cout << "Rotational constants / Hartree:\n" << A << '\t' << B << '\t' << C << '\t' << std::endl;
    std::cout << "kB T / Hartree:\n" << kB * 298 / EHARTREE << std::endl;
    std::cout << "Qapprox = " << sqrt(M_PI)/sigma * sqrt(pow(kB*300/EHARTREE, 3) / (A*B*C)) << std::endl;
    std::cout << "Qvar = " << Q << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << std::endl << duration.count()/1e6 << " secs" << std::endl;
    return 0;
}

MatrixXd getHamiltonian(int lmax, double Ix, double Iy, double Iz) {
    /* Construct the Hamiltonian Matrix
     * Inputs:  lmax;   the maximum quantum number for the spherical basis
     *             I;   the gas-phase moments of inertia [=] amu*Å^2
     * Outputs:    H;   the Hamiltonian matrix (Hermitian) [=] Hartree 
     */
    //std::cout << "NTHREADS =" << omp_get_num_threads() << std::endl;
    //std::cout << "Lmax = " << lmax << std::endl;
    unsigned long long size = ((lmax+1)*(2*lmax+1)*(2*lmax+3)/3.0);
    //std::cout << "Dimensions of the matrix: " << size << std::endl;
    MatrixXd H = MatrixXd::Zero(size, size);

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

    //std::cout << "kappa val is " << kap << "." << std::endl;
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
    //H.print("H = ");
    return H;
}

double getPartitionFunction(double T, MatrixXd& H, int sym) {
    /* Solve the Eigenvalues
     * Inputs:  T;  the temperature [=] K
     *          H;  the Hamiltonian matrix [=] Hartree
     */
    double b = pow(kB * T, -1) * EHARTREE;
    
    // Check Hermiticity
    //if (H.adjoint() == H) {
    //    std::cout << "Hermitian" << std::endl;
    //} else {
    //    std::cout << "Check Hermiticity" << std::endl;
    //    //std::cout << H << std::endl;
    //}
    SelfAdjointEigenSolver<MatrixXd> eigensolver(H);
    VectorXd v = eigensolver.eigenvalues();
    //std::cout << "The eigenvalues of H are " << v.transpose();
    //std::cout << std::endl << "Exponentiated:";
    double Q = (-b*v.array()).exp().sum();
    //std::cout << std::endl << Q << std::endl;
    //for (auto e : eivals.rowwise()) {
    //    std::cout << e << '\t';
    //    Q += exp(-b*e);
    //}
    // Scalar product of temperature scaling
    //MatrixXd bH = -b*H;
    //MatrixXd expbH = expmat_sym(bH);
    //double tr = trace(expbH);
    //std::cout << pow(b,-1) << std::endl;
    //std::cout << std::endl << "Q predicted by eig_sym: " << tr/sym << std::endl;
    return double(Q/sym);
}

std::string getDirectory(std::string sysname) {
    std::string cwd = "/global/scratch/lbettins/rotational-hamiltonian";
    std::string dirname = cwd + "/data";
    if (sysname == "") {
        std::string sysname;
        std::cout << "Enter system name:" << std::endl;
        std::cin >> sysname;
    }
    return dirname+'/'+sysname;
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

//Col<double> getCoefficients(std::string sysname) {
//    std::string dirname = getDirectory(sysname);
//    std::string filename = dirname+'/'+"vdat.txt";
//    std::ifstream is(filename);
//    if (is.fail())
//    {
//        std::cout << "cannot open file " << filename;
//    }
//    double theta, phi, chi, v;
//    std::vector<double> my_vec;
//    while (is) {
//        if (!(is >> theta >> phi >> chi >> v)) {
//            break;
//        }
//        //std::cout << theta << '\t' << phi << '\t' << v << std::endl;
//        my_vec.push_back(v);
//    }
//    Col<double> cvec = conv_to<vec>::from(my_vec);
//    //cvec.print();
//    is.close();
//    return cvec;
//}

