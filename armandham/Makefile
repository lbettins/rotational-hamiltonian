.PHONY : clean

all: ham

ham: hamiltonian.cpp
	$(CXX)	-O2	$(CPPFLAGS)	$^	-o	$@	$(LDFLAGS)

clean:
	-rm ham
