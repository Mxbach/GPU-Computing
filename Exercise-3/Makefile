all: src/task.cu
	nvcc -Werror all-warnings -lcurand -O2 -o task src/task.cu

debug:
	nvcc -g -G -lcurand -o task src/task.cu
	cuda-gdb ./task

clean: 
	rm task
