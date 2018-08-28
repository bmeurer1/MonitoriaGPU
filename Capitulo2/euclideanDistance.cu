#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

/*Função que testa se houve um erro em uma 
chamada a uma função do CUDA previamente.*/
static void cudaTest(const char *msg)
{
  cudaError_t e;

  cudaThreadSynchronize();

  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

__global__ void euclideanDistance(int numPontos, float X, float Y, float Z, float *d_X, float *d_Y, float *d_Z, float *d_distances){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Cada thread calcula a distância do ponto para um único outro ponto do vetor
	if(tid < numPontos){
		d_distances[tid] = sqrt((X - d_X[tid]) * (X - d_X[tid]) + (Y - d_Y[tid]) * (Y - d_Y[tid]) + (Z - d_Z[tid]) * (Z - d_Z[tid]));
	}
}

int main(int argc, char const *argv[]){

	int numPontos   = 0;
	int numThreads = 0;

	if(argc != 3){
		printf("Modo de uso: %s <Numero de Pontos> <Threads Por Bloco>\n", argv[0]);
		exit(-1);
	}else{
		numPontos   = atoi(argv[1]);
		numThreads = atoi(argv[2]);

		if(numThreads > 1024 || numThreads < 1){
			printf("Threads Por Bloco tem que ser entre 1 e 1024\n");
		}
	}

//------------------------------------------------------------------------------------------------------	
//DECLARAÇÃO DE VARIÁVEIS

	float    X,    Y,    Z,   *distances;
	float *h_X, *h_Y, *h_Z, *h_distances;
	float *d_X, *d_Y, *d_Z, *d_distances;

	//Variáveis utilizadas para medir o tempo de execução no device e host
	float tempo_deviceTotal  = 0;
	float tempo_deviceKernel = 0;
	
	clock_t 	h_inicio, h_fim;
	cudaEvent_t d_inicioTotal, d_fimTotal;
	cudaEvent_t d_inicioKernel, d_fimKernel;

	//Cria os eventos de medição de tempo do device
	cudaEventCreate(&d_inicioTotal);
	cudaEventCreate(&d_fimTotal);

	cudaEventCreate(&d_inicioKernel);
	cudaEventCreate(&d_fimKernel);

//------------------------------------------------------------------------------------------------------	
//ALOCAÇÃO E INICIALIZAÇÃO DE VETORES NO HOST

	//Aloca vetor no host
	h_X 	    = (float *) malloc(numPontos * sizeof(float));
	h_Y 	    = (float *) malloc(numPontos * sizeof(float));
	h_Z 	    = (float *) malloc(numPontos * sizeof(float));
	distances   = (float *) malloc(numPontos * sizeof(float));
	h_distances = (float *) malloc(numPontos * sizeof(float));

	//Inicializando oos pontos no host com valores aleatórios
	for(int i = 0; i < numPontos; i++){
		h_X[i] = (float)rand()/(float)RAND_MAX;
		h_Y[i] = (float)rand()/(float)RAND_MAX;
		h_Z[i] = (float)rand()/(float)RAND_MAX;
	}

	X = (float)rand()/(float)RAND_MAX;
	Y = (float)rand()/(float)RAND_MAX;
	Z = (float)rand()/(float)RAND_MAX;

//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO DEVICE

	cudaEventRecord(d_inicioTotal);

	//Aloca vetor no device
	cudaMalloc(&d_X		   , numPontos * sizeof(float)); cudaTest("Falha ao alocar d_X");
	cudaMalloc(&d_Y		   , numPontos * sizeof(float)); cudaTest("Falha ao alocar d_Y");
	cudaMalloc(&d_Z		   , numPontos * sizeof(float)); cudaTest("Falha ao alocar d_Z");
	cudaMalloc(&d_distances, numPontos * sizeof(float)); cudaTest("Falha ao alocar d_distances");

	//Transferindo valores para o device
	cudaMemcpy(d_X, h_X, numPontos * sizeof(float), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_X para d_X");
	cudaMemcpy(d_Y, h_Y, numPontos * sizeof(float), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_Y para d_Y");
	cudaMemcpy(d_Z, h_Z, numPontos * sizeof(float), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_Z para d_Z");

	//Determina quantidade de threads e blocos a serem lançados no kernel 
	dim3 threadsPerBlock(numThreads);
	dim3 numBlocks((numPontos + threadsPerBlock.x - 1)/threadsPerBlock.x);

	cudaEventRecord(d_inicioKernel);
	//Execução do Kernel
	euclideanDistance<<<numBlocks, threadsPerBlock>>>(numPontos, X, Y, Z, d_X, d_Y, d_Z, d_distances);
	cudaEventRecord(d_fimKernel);
	cudaTest("Erro na execução do Kernel");

	//Copia resultados de volta para o host
	cudaMemcpy(h_distances, d_distances, numPontos * sizeof(float), cudaMemcpyDeviceToHost);
	cudaTest("Erro ao copiar d_distances para o host");

	cudaEventRecord(d_fimTotal);

//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO HOST

	h_inicio = clock();
	//Calcula a distancia euclidiana do ponto com relação a todos os outros pontos no vetor
	for(int i = 0; i < numPontos; i++){
		distances[i] = sqrt(pow(X - h_X[i], 2) + pow(Y - h_Y[i], 2) + pow(Z - h_Z[i], 2));
	}
	h_fim = clock();

//------------------------------------------------------------------------------------------------------	
//CHECAGEM DOS RESULTADOS

	int flag = 0;
	for(int i = 0; i < numPontos; i++){
		if(fabs(distances[i] - h_distances[i]) > 0.00001){
			flag = 1;
		}
	}

	if(flag){
		printf("Resultado Incorreto\n");
	}else{
		printf("Resultado Correto\n");
	}

	printf("------------------------------------\n");

//------------------------------------------------------------------------------------------------------	
//CÁLCULO DE TEMPO GASTO NAS EXECUÇÕES

	//Calculo do tempo gasto no host
	float tempo_host = ((float) (h_fim - h_inicio)) / CLOCKS_PER_SEC;
	printf("Tempo CPU\t\t  : %fs\n", tempo_host);

	//Calculo do tempo gasto no device
	cudaEventSynchronize(d_fimTotal);
	cudaEventElapsedTime(&tempo_deviceTotal, d_inicioTotal, d_fimTotal);
	cudaEventElapsedTime(&tempo_deviceKernel, d_inicioKernel, d_fimKernel);

	//A função anterior retorna o tempo em milissegundos.
	//Por isso temos que dividir por 1000
	tempo_deviceTotal = tempo_deviceTotal/1000;
	tempo_deviceKernel = tempo_deviceKernel/1000;

	printf("Tempo GPU Kernel + Memoria: %fs\n", tempo_deviceTotal);
	printf("Tempo GPU Kernel Apenas   : %fs\n", tempo_deviceKernel);
	printf("------------------------------------\n");
	printf("Speed Up Kernel + Memoria : %lf \n", tempo_host/tempo_deviceTotal);
	printf("Speed Up Kernel Apenas    : %lf \n", tempo_host/tempo_deviceKernel);

//------------------------------------------------------------------------------------------------------	
//LIBERA MEMÓRIA E RESETA GPU

	free(h_X);
	free(h_Y);
	free(h_Z);
	free(distances);
	free(h_distances);

	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_Z);
	cudaFree(d_distances);

	cudaEventDestroy(d_inicioTotal);
	cudaEventDestroy(d_fimTotal);
	cudaEventDestroy(d_inicioKernel);
	cudaEventDestroy(d_fimKernel);

	cudaDeviceReset();

	return 0;
}