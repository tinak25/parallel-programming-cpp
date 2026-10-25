CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pthread
INCLUDES := -Iinclude
HEADERS  := $(wildcard include/*.hpp)
BIN := bin

.PHONY: all cpu mpi data check clean
all: cpu mpi
cpu: $(BIN)/sequential $(BIN)/threads
mpi: $(BIN)/mpi

$(BIN):
	mkdir -p $(BIN)
$(BIN)/sequential: src/sequential.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
$(BIN)/threads: src/threads.cpp $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
$(BIN)/mpi: src/mpi.cpp $(HEADERS) | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

# A bigger file for benchmarking (30 days x 60 lights, ~520k readings, not committed)
data: data/month.csv
data/month.csv:
	python3 tests/generate_data.py --days 30 --lights 60 --seed 2 --out $@

check: all
	./tests/check.sh

clean:
	rm -rf $(BIN)
