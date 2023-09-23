# Compiler and compiler flags
CXX := g++
CXXFLAGS := -std=c++11 -Wall -Weffc++ -Wextra -Wconversion -Wshadow -I. -I./include/

TARGET = autotest

all: $(TARGET)

$(TARGET): unittest.o Deep/node.o Deep/nn.o Deep/utility.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) unittest.o Deep/node.o Deep/nn.o Deep/utility.o

unittest.o: unittest.cpp $(wildcard Deep/*.h)
	$(CXX) $(CXXFLAGS) -c unittest.cpp

Deep/utility.o: Deep/utility.h Deep/node.h

Deep/node.o: Deep/node.h

Deep/nn.o: Deep/nn.h Deep/base.h

.PHONY: clean
clean:
	rm Deep/*.o *.o *.exe