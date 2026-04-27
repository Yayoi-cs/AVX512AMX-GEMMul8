CXX      ?= g++
NVCC     ?= nvcc
OPTLEVEL ?= -O3
CXXFLAGS ?= $(OPTLEVEL) -std=c++17 -Wall -Wextra -Wno-unused-parameter \
            -march=native -mtune=native -fopenmp -fno-math-errno -mprefer-vector-width=512
NVCCFLAGS ?= $(OPTLEVEL) -std=c++17 -Xcompiler "-march=native -mtune=native -fopenmp"
INCLUDES := -Iinclude -Isrc
MKL_PKG ?= mkl-sdl

LIB := lib/libgemmul8_avx512.a
OBJS := src/gemmul8_avx512.o
DEPS := include/gemmul8_avx512.hpp $(wildcard src/*.hpp)

all: $(LIB)

lib:
	@mkdir -p lib

$(LIB): lib $(OBJS)
	ar rcs $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

src/gemmul8_avx512.o: $(DEPS)

clean:
	rm -f $(OBJS) $(LIB)
	rmdir lib 2>/dev/null || true

.PHONY: all clean lib
