.PHONY : clean

#all: gaunt	ham
#
#gaunt: gaunt_coeffs.cpp
#	$(CXX)	$(CPPFLAGS)	$^	-o	$@	$(LDFLAGS)
all: ham

ham: hamiltonian.cpp
	$(CXX)	-O2	$(CPPFLAGS)	$^	-o	$@	$(LDFLAGS)

clean:
	-rm gaunt ham
