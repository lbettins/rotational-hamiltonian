CPP = /usr/local/opt/llvm/bin/clang++
CPPFLAGS = -I/usr/local/opt/llvm/include -I/usr/local/opt/armadillo/include -I/Users/lancebettinson/Thesis/umrr/code/wigner-cpp/include -fopenmp -O2 -std=c++20
LDFLAGS = -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -DARMA_DONT_USE_WRAPPER -framework Accelerate

all: gaunt	ham

gaunt: gaunt_coeffs.cpp
	$(CPP)	$(CPPFLAGS)	$^	-o	$@	$(LDFLAGS)

ham: hamiltonian.cpp
	$(CPP)	$(CPPFLAGS)	$^	-o	$@	$(LDFLAGS)
