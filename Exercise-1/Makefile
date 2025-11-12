SRC_DIR=src

build:
# 	clang++ -Wall -Wextra -Werror -O2 -o slow slow.cpp
	g++ -Wall -Wextra -Werror -O2 -o slow $(SRC_DIR)/slow.cpp

cuda:
	nvcc -O2 -o acc $(SRC_DIR)/accelerated.cu

clean: 
	rm -f acc
	rm -f slow