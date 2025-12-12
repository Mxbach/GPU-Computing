all: cuda cpu

cuda:
	nvcc -o task src/task.cu

debug:
	nvcc -g -G -lcurand -o task src/task.cu

cpu: 
	g++ -o cpu src/cpu.cpp

clean:
	rm -f cpu
	rm -f task

.PHONY: all cuda cpu clean debug