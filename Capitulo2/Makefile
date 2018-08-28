all: saxpy eucDist dotProduct vectorSum anyDimMatMul sharedMemMatMul

saxpy: saxpy.cu
	nvcc -o saxpy saxpy.cu

eucDist: Ex1.cu
	nvcc -o ex1 Ex1.cu

dotProduct: Ex2.cu
	nvcc -o ex2 Ex2.cu

vectorSum: Ex3.cu
	nvcc -o ex3 Ex3.cu

anyDimMatMul: Ex4.cu
	nvcc -o ex4 Ex4.cu

sharedMemMatMul: Ex5.cu
	nvcc -o ex5 Ex5.cu

clean: 
	rm saxpy ex1 ex2 ex3 ex4 ex5