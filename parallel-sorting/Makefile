CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pthread
INCLUDES := -Iinclude
HEADERS  := $(wildcard include/*.hpp)

ifeq ($(shell uname -s),Darwin)
  CL_LIBS ?= -framework OpenCL
else
  CL_LIBS ?= -lOpenCL
endif
CL_CFLAGS ?=

BIN := bin
.PHONY: all cpu mpi opencl check clean
all: cpu mpi opencl
cpu: $(BIN)/sequential $(BIN)/threads
mpi: $(BIN)/mpi
opencl: $(BIN)/mpi_opencl

$(BIN):
	mkdir -p $(BIN)

$(BIN)/sequential: src/sequential.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
$(BIN)/threads: src/threads.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
$(BIN)/mpi: src/mpi.cpp $(HEADERS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
$(BIN)/mpi_opencl: src/mpi_opencl.cpp $(HEADERS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(CL_CFLAGS) $(INCLUDES) $< -o $@ $(CL_LIBS)

check: all
	./tests/check.sh

clean:
	rm -rf $(BIN)
