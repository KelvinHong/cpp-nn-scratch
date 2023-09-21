# Compiler and compiler flags
CXX := g++
CXXFLAGS := -std=c++14 -Wall -Weffc++ -Wextra -Wconversion -Wshadow -I.

TARGET = main

all: $(TARGET)

$(TARGET): unittest.o Deep/node.o Deep/nn.o
	$(CXX) $(CXXFLAGS) -o main unittest.o Deep/node.o Deep/nn.o

unittest.o: unittest.cpp Deep/node.h Deep/nn.h Deep/base.h
	$(CXX) $(CXXFLAGS) -c unittest.cpp

Deep/node.o: Deep/node.h

Deep/nn.o: Deep/nn.h Deep/base.h

.PHONY: clean
clean:
	rm Deep/*.o *.o