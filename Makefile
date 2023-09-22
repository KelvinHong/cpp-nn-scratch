# Compiler and compiler flags
CXX := g++
CXXFLAGS := -std=c++11 -Wall -Weffc++ -Wextra -Wconversion -Wshadow -I.

TARGET = autotest

all: $(TARGET)

$(TARGET): unittest.o Deep/node.o Deep/nn.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) unittest.o Deep/node.o Deep/nn.o

unittest.o: unittest.cpp $(wildcard Deep/*.h)
	$(CXX) $(CXXFLAGS) -c unittest.cpp

Deep/node.o: Deep/node.h

Deep/nn.o: Deep/nn.h Deep/base.h

.PHONY: clean
clean:
	rm Deep/*.o *.o *.exe