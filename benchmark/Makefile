# Benchmark for Ozaki Scheme II CPU implementations (AVX-512, AMX, MKL).
#
# Builds ./benchmark linking the prebuilt static libs from ../avx-512 and
# ../amx, plus Intel MKL for the cblas_dgemm baseline.

CXX      ?= g++
OPTLEVEL ?= -O3
CXXFLAGS ?= $(OPTLEVEL) -std=c++17 -Wall -Wextra \
            -march=native -mtune=native -mamx-tile -mamx-int8 \
            -fopenmp -fno-math-errno -mprefer-vector-width=512

MKL_PKG    ?= mkl-sdl
MKL_CFLAGS := $(shell pkg-config --cflags $(MKL_PKG) 2>/dev/null)
MKL_LIBS   := $(shell pkg-config --libs   $(MKL_PKG) 2>/dev/null)

AVX_LIB := ../avx-512/lib/libgemmul8_avx512.a
AMX_LIB := ../amx/lib/libgemmul8_amx.a

INCLUDES := -I../avx-512/include -I../amx/include

BIN := benchmark

all: $(BIN)

$(AVX_LIB):
	$(MAKE) -C ../avx-512 lib/libgemmul8_avx512.a

$(AMX_LIB):
	$(MAKE) -C ../amx lib/libgemmul8_amx.a

$(BIN): benchmark.cpp $(AVX_LIB) $(AMX_LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(MKL_CFLAGS) \
	    -o $@ benchmark.cpp $(AVX_LIB) $(AMX_LIB) $(MKL_LIBS) -lm

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: all run clean
