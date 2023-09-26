# Compiler and compiler flags
CXX := g++
CXXFLAGS := -std=c++11 -Wall -Weffc++ -Wextra -Wconversion -Wshadow -I. -I./include/

TARGET = autotest
MODEL1 = firstModel

all: $(TARGET) $(MODEL1)

$(MODEL1): tests/regressionTest.o Deep/node.o Deep/nn.o Deep/utility.o Deep/base.o Deep/optimizer.o
	$(CXX) $(CXXFLAGS) -o $(MODEL1) $^

$(TARGET): tests/unittest.o Deep/node.o Deep/nn.o Deep/utility.o Deep/base.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) $^

tests/regressionTest.o: tests/regressionTest.cpp $(wildcard Deep/*.h)
	$(CXX) $(CXXFLAGS) -c $< -o $@

tests/unittest.o: tests/unittest.cpp $(wildcard Deep/*.h)
	$(CXX) $(CXXFLAGS) -c $< -o $@

Deep/optimizer.o: Deep/optimizer.h Deep/base.h

Deep/utility.o: Deep/utility.h Deep/node.h

Deep/nn.o: Deep/nn.h Deep/base.h

Deep/base.o: Deep/base.h Deep/node.h

Deep/node.o: Deep/node.h

.PHONY: clean
clean:
	-rm Deep/*.o *.o *.exe tests/*.o $(TARGET)