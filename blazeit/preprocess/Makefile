#all:
	# g++ -O3 -std=c++14 -fopenmp seqnms.cpp -o seqnms
#	g++ -g -O3 -std=c++14 -fopenmp filter_cc.cpp clipper.cpp -o filter_cc

CXX=g++
CXXFLAGS = -std=c++14 -O3 -fopenmp
LDFLAGS = -lglog -ldouble-conversion -lfolly -L/lfs/1/ddkang/local/lib

all: seqnms filter_cc

clipper.o: clipper.hpp

seqnms: seqnms.o
	$(CXX) $(CXXFLAGS) -o seqnms seqnms.o

filter_cc: filter_cc.o clipper.o clipper.hpp
	$(CXX) $(CXXFLAGS) -o filter_cc filter_cc.o clipper.o $(LDFLAGS)

