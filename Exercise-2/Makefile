build: src/slow_scan.cu
	nvcc -Werror all-warnings -lcurand -O2 -o slow_scan src/slow_scan.cu

debug:
	nvcc -g -G -lcurand -o slow_scan src/slow_scan.cu

clean:
	rm -f slow_scan