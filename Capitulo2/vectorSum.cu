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

//Primeiro kernel carrega o valor a ser somado
__global__ void loadValues(double *vetorA, double *aux, int salto){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid >= salto){
		aux[tid] = vetorA[tid - salto];
	}

}

//Segundo kernel efetua a soma
__global__ void sumValues(double *vetorA, double *aux, int salto){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid >= salto){
		vetorA[tid] = aux[tid] + vetorA[tid];
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

	double *h_vectorA, *h_vectorB, resultCPU = 0;
	double *d_vectorA, *d_aux,     resultGPU = 0;

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
//ALOCAÇÃO E INICIALIZAÇÃO DE VETORES NO HOST E NO DEVICE


	//Aloca vetor no host
	h_vectorA = (double *) malloc(numPontos * sizeof(double));
	h_vectorB = (double *) malloc(numPontos * sizeof(double));

	//Inicializando oos pontos no host com valores aleatórios
	for(int i = 0; i < numPontos; i++){
		h_vectorA[i] = (double)rand()/(double)RAND_MAX;
	}

//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO DEVICE


	cudaEventRecord(d_inicioTotal);

	//Aloca vetor no device
	cudaMalloc(&d_vectorA, numPontos * sizeof(double)); cudaTest("Falha ao alocar d_vector");
	cudaMalloc(&d_aux    , numPontos * sizeof(double)); cudaTest("Falha ao alocar d_vector");

	cudaMemcpy(d_vectorA, h_vectorA, numPontos * sizeof(double), cudaMemcpyHostToDevice);
	cudaTest("Falha ao copiar h_vectorA para d_vectorA");

	//Determina quantidade de threads e blocos a serem lançados no kernel
	dim3 threadsPerBlock(numThreads);
	dim3 numBlocks((numPontos + threadsPerBlock.x - 1)/threadsPerBlock.x);

	cudaEventRecord(d_inicioKernel);

	//Dividimos o kernel em dois para obtermos sincronização entre blocos
	for(int salto = 1; salto < numPontos; salto *= 2){

		loadValues<<<numBlocks, threadsPerBlock>>>(d_vectorA, d_aux, salto);
		cudaTest("Erro ao executar o Kernel");

		sumValues<<<numBlocks, threadsPerBlock>>>(d_vectorA, d_aux, salto);
		cudaTest("Erro ao executar o Kernel");
	
	}

	cudaEventRecord(d_fimKernel);

	cudaMemcpy(h_vectorB, d_vectorA, numPontos* sizeof(double), cudaMemcpyDeviceToHost);

	resultGPU = h_vectorB[numPontos - 1];

	cudaEventRecord(d_fimTotal);


//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO HOST

	h_inicio = clock();
	for(int i = 1; i < numPontos; i++){

		h_vectorA[i] += h_vectorA[i - 1];
	}

	resultCPU = h_vectorA[numPontos - 1];
	h_fim = clock();

//------------------------------------------------------------------------------------------------------	
//EXIBIÇÃO DOS RESULTADOS

	printf("Resultado CPU: %lf\n", resultCPU);
	printf("Resultado GPU: %lf\n", resultGPU);
	printf("------------------------------------\n");

//------------------------------------------------------------------------------------------------------	
//CÁLCULO DE TEMPO GASTO NAS EXECUÇÕES

	//Calculo do tempo gasto no host
	double tempo_host = ((double) (h_fim - h_inicio)) / CLOCKS_PER_SEC;
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

	free(h_vectorA);
	free(h_vectorB);

	cudaFree(d_vectorA);

	cudaEventDestroy(d_inicioTotal);
	cudaEventDestroy(d_fimTotal);
	cudaEventDestroy(d_inicioKernel);
	cudaEventDestroy(d_fimKernel);

	cudaDeviceReset();

	return 0;
}