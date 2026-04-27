CXX      ?= g++
NVCC     ?= nvcc
OPTLEVEL ?= -O3
CXXFLAGS ?= $(OPTLEVEL) -std=c++17 -Wall -Wextra -Wno-unused-parameter \
            -march=native -mtune=native -mamx-tile -mamx-int8 \
            -fopenmp -fno-math-errno -mprefer-vector-width=512
NVCCFLAGS ?= $(OPTLEVEL) -std=c++17 -Xcompiler "-march=native -mtune=native -mamx-tile -mamx-int8 -fopenmp"
INCLUDES := -Iinclude -Isrc

LIB := lib/libgemmul8_amx.a
OBJS := src/gemmul8_amx.o
DEPS := include/gemmul8_amx.hpp $(wildcard src/*.hpp)

all: $(LIB)

lib:
	@mkdir -p lib

$(LIB): lib $(OBJS)
	ar rcs $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

src/gemmul8_amx.o: $(DEPS)

clean:
	rm -f $(OBJS) $(LIB)
	rmdir lib 2>/dev/null || true

.PHONY: all clean run lib
