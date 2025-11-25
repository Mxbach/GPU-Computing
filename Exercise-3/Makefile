all:
	nvcc -Werror all-warnings -lcurand -O2 -o task task.cu

debug:
	nvcc -g -G -lcurand -o task task.cu

clean: 
	rm task
