CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pthread
OMPFLAGS ?= -fopenmp
INCLUDES := -Iinclude
HEADERS  := $(wildcard include/*.hpp)

ifeq ($(shell uname -s),Darwin)
  CL_LIBS ?= -framework OpenCL
else
  CL_LIBS ?= -lOpenCL
endif
CL_CFLAGS ?=

BIN := bin
CPU := $(BIN)/sequential $(BIN)/sequential_O0 $(BIN)/pthreads $(BIN)/openmp
MPI := $(BIN)/mpi $(BIN)/mpi_openmp
CL  := $(BIN)/opencl $(BIN)/mpi_opencl

.PHONY: all cpu mpi opencl check clean
all: cpu mpi opencl
cpu: $(CPU)
mpi: $(MPI)
opencl: $(CL)

$(BIN):
	mkdir -p $(BIN)

$(BIN)/sequential: src/sequential.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
$(BIN)/sequential_O0: src/sequential.cpp $(HEADERS) | $(BIN)
	$(CXX) -O0 -std=c++17 -pthread $(INCLUDES) $< -o $@
$(BIN)/pthreads: src/pthreads.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
$(BIN)/openmp: src/openmp.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) $< -o $@
$(BIN)/mpi: src/mpi.cpp $(HEADERS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
$(BIN)/mpi_openmp: src/mpi_openmp.cpp $(HEADERS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) $< -o $@
$(BIN)/opencl: src/opencl.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(CL_CFLAGS) $(INCLUDES) $< -o $@ $(CL_LIBS)
$(BIN)/mpi_opencl: src/mpi_opencl.cpp $(HEADERS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(CL_CFLAGS) $(INCLUDES) $< -o $@ $(CL_LIBS)

# Every version must produce exactly the same matrix (see tests/check.sh)
check: all
	./tests/check.sh

clean:
	rm -rf $(BIN)
