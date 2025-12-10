CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude

SRC = \
  src/main.cpp \
  src/Network.cpp \
  src/MNIST.cpp

OBJ = $(SRC:.cpp=.o)
BIN = MLP

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)
