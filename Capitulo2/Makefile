all: saxpy eucDist dotProduct vectorSum anyDimMatMul sharedMemMatMul

saxpy: saxpy.cu
	nvcc -o saxpy saxpy.cu

eucDist: euclideanDistance.cu
	nvcc -o euclideanDistance euclideanDistance.cu

dotProduct: dotProduct.cu
	nvcc -o dotProduct dotProduct.cu

vectorSum: vectorSum.cu
	nvcc -o vectorSum vectorSum.cu

anyDimMatMul: anyDimMatMult.cu
	nvcc -o anyDimMatMul anyDimMatMult.cu

sharedMemMatMul: sharedMemMatMult.cu
	nvcc -o sharedMemMatMul sharedMemMatMult.cu

clean: 
	rm saxpy euclideanDistance dotProduct vectorSum anyDimMatMul sharedMemMatMul