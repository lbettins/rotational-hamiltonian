CPP = /usr/local/opt/llvm/bin/clang++
CPPFLAGS = -I/usr/local/opt/llvm/include -I/usr/local/opt/armadillo/include -I/usr/local/include -fopenmp -O2 -std=c++20
LDFLAGS = -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -L/usr/local/lib -L/usr/local/opt/armadillo/lib -lwignerSymbols -larmadillo -framework Accelerate
.PHONY : clean

all: ham

ham: hamiltonian.cpp
	$(CPP)	-O2	$(CPPFLAGS)	$^	-o	$@	$(LDFLAGS)

clean:
	-rm ham
