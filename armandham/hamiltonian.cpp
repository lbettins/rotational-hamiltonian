#define _USE_MATH_DEFINES

#include <fstream>
#include <iostream>
#include <iomanip>  // std::setw
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <string>
#include <armadillo>
#include <vector>
#include "wignerSymbols.h"
//#include "wigner/gaunt.hpp"

using namespace std;
using namespace arma;

static const double kB = 1.3806504e-23;         // J/K
static const double NA = 6.02214179e23;         // 1/mol
static const double EHARTREE = 4.35974434e-18;  // J/Hartree
static const double AMU = 1.660538921e-27;      // kg/amu
static const double HBAR = 1.054571726e-34;     // J.s
static const double HBAR1 = HBAR / EHARTREE;    // Hartree.s
static const double HBAR2 = HBAR * 1e20 / AMU;  // amu.Å^2/s
static const double HARTREE2KCALMOL = EHARTREE*NA/4184; // kcal/mol
static const double SCH4 = 186.25; // J/mol.K

Mat<complex<double>> getHamiltonian(int lmax, double Ix, double Iy, double Iz, vector<complex<double>>& a, int seriesLmax);
SpMat<double> getSparseHam(int lmax, double Ix, double Iy, double Iz, vector<complex<double>>& a);
//Col<double> getCoefficients(string sysname);
vector<complex<double>> getWignerCoeffs(string sysname, bool freeRotor);
vector<double> getMomentOfInertia(string sysname="METH-CHA");
double getSymNum(string sysname);
int getExpansionLmax(string sysname, bool freeRotor);
string getDirectory(string sysname);
Col<double> getEnergyEigvals(Mat<complex<double>>& H);
vector<double> getThermo(double T, Col<double>& eigvals, double sym=1);
//double getPartitionFunction(double T, Mat<complex<double>>& H, double sym=1);
double getSparseQ(double T, SpMat<double>& H, int sym=1);

int main(int argc, char** argv) {
    /* argv[0]  Program name
     * argv[1]  System name
     * argv[2]  Lmax (start)
     * argv[3]  free-rotor
     * argv[4]  temperature / K
     */
    if (argc != 5) {
        throw runtime_error("Enter systemName as it appears in data directory, Lmax, free-rotor flag, temperature in Kelvin");
    }

    /* Extract molecular information */
    string sysname = argv[1];
    bool freeRotor = stoi(argv[3]);
    double T = stod(argv[4]);
    string dirname = getDirectory(sysname);
    cout << "Directory containing data is: " << dirname << endl;
    vector<complex<double>> a = getWignerCoeffs(sysname, freeRotor);
    double sigma = getSymNum(sysname);
    int seriesLmax = getExpansionLmax(sysname, freeRotor);
    cout << "Symmetry number for system " << sysname << " is " <<  sigma << '.' << endl;
    cout << "Expansion series has an LMax of " << seriesLmax << '.' << endl;

    /* Moments of Inertia */
    vector<double> Ivec = getMomentOfInertia(sysname);
    cout << "Moments of Inertia:" << endl;
    for (double i : Ivec) {
        cout << i << '\t';
    }
    cout << endl;
 
    /*
     *  Rotational Constants + Classical Partition Function
     */
    double B = HBAR1*HBAR2/(2.0*Ivec[2]);
    double A = HBAR1*HBAR2/(2.0*Ivec[1]);
    double C = HBAR1*HBAR2/(2.0*Ivec[0]);
    //cout << "kB T / Hartree:\n" << kB * 298 / EHARTREE << endl;
    cout << "Qapprox = " << sqrt(M_PI)/sigma * sqrt(pow(kB*T/EHARTREE, 3) / (A*B*C)) << endl;  

    /*
     *  Dense Matrix Implementation 
     */
    bool converge = false;
    bool dense = true;
    int lmax = atoi(argv[2]);
    double Q;
    double U;
    double S;
    double Qprev = 0;
    vector<double> thermo;
    cout << left << setw(15) << "Lmax" << setw(15) << "Q" << setw(15) << "U [kcal/mol]" << setw(15) << "S [cal/mol.K]" << setw(20) <<
        "Constr.Time [s]" << setw(16) << "Diag.Time [s]" << setw(16) << "Time [s]" << setw(16) << "dQ" << endl;
    auto start = chrono::high_resolution_clock::now();
    auto qmid = chrono::high_resolution_clock::now();
    do {
        auto qstart = chrono::high_resolution_clock::now();
        if (dense) {
            /****  Dense Matrix Implementation  ****/
            Mat<complex<double>> H = getHamiltonian(lmax, Ivec[0], Ivec[1], Ivec[2], a, seriesLmax);
            qmid = chrono::high_resolution_clock::now();
            if (!H.is_hermitian(1e-5)) {
                cout << "NOT HERMITIAN:" <<  endl;
                H.brief_print("H = ");
            }
            //Q = getPartitionFunction(T, H, sigma);
            Col<double> eigvals = getEnergyEigvals(H);
            thermo = getThermo(T, eigvals, sigma);
            Q = thermo[0];
            U = thermo[1];
            S = thermo[2];
        } else {
            /****  Sparse Matrix Implementation  ****/
            SpMat<double> H = getSparseHam(lmax, Ivec[0], Ivec[1], Ivec[2], a);
            Q = getSparseQ(T, H, sigma);
        }
        auto qend = chrono::high_resolution_clock::now();
        auto qmatDur = chrono::duration_cast<chrono::microseconds>(qmid - qstart);
        auto qdiagDur = chrono::duration_cast<chrono::microseconds>(qend - qmid);
        auto qduration = chrono::duration_cast<chrono::microseconds>(qend - qstart);
        double dQ = fabs(Q-Qprev);
        cout << left << setw(15) << lmax << setw(15) << Q << setw(15) << U << setw(15) << S << setw(20) << qmatDur.count()/1e6 << setw(16) 
            << qdiagDur.count()/1e6 << setw(16) <<
            qduration.count()/1e6 << setw(16) << dQ << endl;
        if (dQ < 1e-5 && lmax > 20) {
            converge = true;
            cout << "DeltaQ = " << fabs(Q-Qprev) << endl;
            cout << "Convergence criterion met!" << endl;
            cout << "Lmax = " << lmax << endl;
        }
        lmax++;
        Qprev = Q;
    } while (!converge);
    cout << endl;
    cout << endl;
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    cout << endl << duration.count()/1e6 << " secs" << endl;
    return 0;
}

Col<double> getEnergyEigvals(Mat<complex<double>>& H) {
    Col<double> eigvals = eig_sym(H);
    return eigvals;
}

vector<double> getThermo(double T, Col<double>& eigvals, double sym) {
    /* Solve the relevant thermodynamics 
     * Inputs:  T;          the temperature [=] K
     *          eigvals;    the energy microstates
     *          sym;        the symmetry number
     * Outputs: a vector containing
     *          v[0];       the partition function
     *          v[1];       the internal energy [=] kcal/mol
     *          v[2];       the entropy [=] cal/mol.K
     */
    double b = pow(kB * T, -1) * EHARTREE;
    double Q = 0;
    double U = 0;
    for (double e : eigvals) {
        U += e*exp(-b*e);
        Q += exp(-b*e);
    }
    U /= Q;
    Q /= sym;
    double F = -pow(b, -1) * log(Q);
    double S = (U-F)/T;
    U *= HARTREE2KCALMOL;
    S *= HARTREE2KCALMOL*1000;
    vector<double> v {Q, U, S};
    return v;
}

Mat<complex<double>> getHamiltonian(int lmax, double Ix, double Iy, double Iz, vector<complex<double>>& a, int seriesLmax) {
    /* Construct the Hamiltonian Matrix
     * Inputs:  lmax;   the maximum quantum number forthe spherical basis
     *             I;   the gas-phase moments of inertia [=] amu*Å^2
     *             a;   coefficients for potential in the Wigner D Matrix basis
     * Outputs:    H;   the Hamiltonian matrix (Hermitian) [=] Hartree 
     */
    long double size = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3.0;
    Mat<complex<double>> H = zeros<cx_mat>(size, size);

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
                    unsigned long long j = 0;
                    for (int ell = 0; ell < lmax+1; ell++) {
                        for (int mm = -ell; mm <= ell; mm++) {
                            for (int kk = -ell; kk <= ell; kk++) {
                                //unsigned long long j = (4*ell*ell*ell/3.0) + 2*ell*ell + (5*ell/3.0) + 2*mm*ell + mm + kk;
                                if (j > i) {
                                    j++;
                                    continue;
                                }
                                complex<double> Vij (0,0);
                                if (seriesLmax) {
                                    // Solve Potential Component of Matrix by
                                    // iterating over Clebsch-Gordan coefficients (if applicable)
                                    int ind = 0;
                                    double sign = pow(-1.0, mm+kk);
                                    for (int L = 0; L < seriesLmax+1; L++) {
                                        for (int M = -L; M <= L; M++) {
                                            double Wlm = WignerSymbols::wigner3j(L,el,ell,M,m,-mm);
                                            for (int K = -L; K <= L; K++) {
                                                if (Wlm == 0) {
                                                    ind++;
                                                    continue;
                                                }
                                                double Wlk = WignerSymbols::wigner3j(L,el,ell,K,k,-kk);
                                                double val = 8*M_PI*M_PI*sign*Wlm*Wlk;
                                                Vij += a.at(ind) * val;
                                                ind++;
                                            }
                                        }
                                    }
                                    if (ind != a.size()) {
                                        cout << "LOOP ABORTED BEFORE END OF ARRAY" << endl;
                                    }
                                }
                                // Solve the Kinetic Component resulting from raising and lowering operators
                                // (and add the potential component to the diagonal)
                                //Vij *= 0.1;
                                if (i == j) {
                                    H(i,j) += complex<double> (0.5*(A+C)*el*(el+1) + 0.5*(A-C)*kap*k*k + Vij.real(), 0);
                                    if (k+2 <= el) {
                                        complex<double> val (0.25*(C-A)*sqrt(el*(el+1)-k*(k+1))*sqrt(el*(el+1)-(k+1)*(k+2)), 0);
                                        H(i+2,j) += val;
                                        H(j,i+2) += val;
                                    }
                                } else {
                                    H(i,j) += Vij;
                                    H(j,i) += conj(Vij);
                                }
                                //cout << "< " << i << "|V|" << j << " >" << "\t=\t" << Vij << '\t' << conj(Vij) << endl;
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
    //cout << "Constructed matrix with LMAX " << lmax << '.' << endl;
    return H;
}

double getSymNum(string sysname) {
    string dirname = getDirectory(sysname);
    string filename = dirname+"/s.txt";
    ifstream is(filename);
    double val;
    is >> val;
    return val;
}

int getExpansionLmax(string sysname, bool freeRotor) {
    if (freeRotor) return 0;
    string dirname = getDirectory(sysname);
    string filename = dirname+"/lmax.txt";
    ifstream is(filename);
    int val;
    is >> val;
    return val;
}

vector<complex<double>> getWignerCoeffs(string sysname, bool freeRotor) {
    vector<complex<double>> a;
    if (freeRotor) {
        return a;
    }
    string dirname = getDirectory(sysname);
    string filename = dirname+"/aimag.txt";
    ifstream is(filename);
    complex<double> val;
    while (is) {
        if(!(is >> val)) {
            break;
        }
        a.push_back(val);
    }
    return a;
}

vector<double> getMomentOfInertia(string sysname) {
    string dirname = getDirectory(sysname);
    string filename = dirname+'/'+"I.txt";
    ifstream is(filename);
    double val;
    vector<double> Ivec;
    while (is) {
        if (!(is >> val)) {
            break;
        }
        Ivec.push_back(val);
    }
    return Ivec;
}

string getDirectory(string sysname) {
    //string dirname = "/Users/lancebettinson/Thesis/umrr/code/hamiltonian-cpp/data";
    string dirname = "/global/scratch/lbettins/rotational-hamiltonian/data";
    if (sysname == "") {
        string sysname;
        cout << "Enter system name:" << endl;
        cin >> sysname;
    }
    return dirname+'/'+sysname;
}

SpMat<double> getSparseHam(int lmax, double Ix, double Iy, double Iz, vector<complex<double>>& a) {
    /* Construct the Hamiltonian Matrix
     * Inputs:  lmax;   the maximum quantum number forthe spherical basis
     *             I;   the gas-phase moments of inertia [=] amu*Å^2
     * Outputs:    H;   the Hamiltonian matrix (Hermitian) [=] Hartree 
     */
    unsigned long long size = (lmax+1)*(2*lmax+1)*(2*lmax+3)/3.0;
    SpMat<double> H = sp_mat(size, size);

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
                                    } catch (const exception& e) {
                                        cout << "Failure at index: " <<
                                            i << "\t(" << el << ',' << m << ',' <<
                                            k << ')' << endl;
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

double getSparseQ(double T, SpMat<double>& H, int sym) {
    /* Solve the Eigenvalues forSparse Matrix
     * Inputs:  T;  the temperature [=] K
     *          H;  the (sparse) Hamiltonian matrix [=] Hartree
     */
    double b = pow(kB * T, -1) * EHARTREE;
    vec eigval;
    mat eigvec;
    //cout << "Number of rows: " << H.n_rows << endl;
    eigs_sym(eigval, eigvec, H, H.n_rows-1);
    double Q = 0;
    //cout << "Eigenvalues: " << endl;
    for (double e : eigval) {
        //cout << e << '\t';
        Q += exp(-b * e);
    }
    //cout << "Q predicted by eigs_sym: " << Q/sym << endl;
    return double(Q/sym);
}

Col<double> getCoefficients(string sysname) {
    string dirname = getDirectory(sysname);
    string filename = dirname+'/'+"vdat.txt";
    ifstream is(filename);
    if (is.fail())
    {
        cout << "cannot open file " << filename;
    }
    double theta, phi, v;
    vector<complex<double>> my_vec;
    while (is) {
        if (!(is >> theta >> phi >> v)) {
            break;
        }
        //cout << theta << '\t' << phi << '\t' << v << endl;
        my_vec.push_back(v);
    }
    Col<double> cvec = conv_to<vec>::from(my_vec);
    //cvec.print();
    is.close();
    return cvec;
}
