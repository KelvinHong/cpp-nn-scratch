# Compiler and compiler flags
CXX := g++
CXXFLAGS := -std=c++11 -Wall -Weffc++ -Wextra -Wconversion -Wshadow -I. -I./include/

TARGET = autotest

all: $(TARGET)

$(TARGET): tests/unittest.o Deep/node.o Deep/nn.o Deep/utility.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) tests/unittest.o Deep/node.o Deep/nn.o Deep/utility.o

tests/unittest.o: tests/unittest.cpp $(wildcard Deep/*.h)
	$(CXX) $(CXXFLAGS) -c tests/unittest.cpp -o $@

Deep/utility.o: Deep/utility.h Deep/node.h

Deep/node.o: Deep/node.h

Deep/nn.o: Deep/nn.h Deep/base.h

.PHONY: clean
clean:
	rm Deep/*.o *.o *.exe $(TARGET)