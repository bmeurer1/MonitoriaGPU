#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

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

__global__ void multiplyElements(int tamVetor, double * vetorA, double *vetorB){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Faz a multiplicão dos elementos de cada posição dos vetores e guarda
	//no vetor A para ser usado no próximo Kernel
	for(int i = tid; i < tamVetor; i += gridDim.x * blockDim.x){
		vetorA[i] = vetorA[i] * vetorB[i];
	}
}

__global__ void vectorSum(double *vetorA, double *vetorB){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for(int salto = 1; salto < blockDim.x; salto *= 2){
		double aux;

		if(threadIdx.x >= salto){
			aux = vetorA[tid - salto];
		}

		__syncthreads();

		if(threadIdx.x >= salto){
			vetorA[tid] = aux + vetorA[tid];
		}

		__syncthreads();
	}

	if(threadIdx.x == blockDim.x - 1){
		vetorB[blockIdx.x] = vetorA[tid];
	}
}

int main(int argc, char const *argv[]){

	int tamVetor   = 0;
	int numThreads = 0;

	if(argc != 3){
		printf("Modo de uso: %s <Tamanho dos Vetores> <Threads Por Bloco>\n", argv[0]);
		exit(-1);
	}else{
		tamVetor   = atoi(argv[1]);
		numThreads = atoi(argv[2]);

		if(numThreads > 1024 || numThreads < 1){
			printf("Threads Por Bloco tem que ser entre 1 e 1024\n");
		}
	}

//------------------------------------------------------------------------------------------------------	
//DECLARAÇÃO DE VARIÁVEIS

	double *h_vetorA, *h_vetorB, *h_vetorC, h_resultado = 0;
	double *d_vetorA, *d_vetorB, *d_vetorC, d_resultado = 0;

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

	//Determina quantidade de threads e blocos a serem lançados no kernel
	//para calcular o tamanho dos vetores 
	dim3 threadsPerBlock(numThreads);
	dim3 numBlocks((tamVetor + threadsPerBlock.x - 1)/threadsPerBlock.x);

	//Aloca vetor no host
	h_vetorA = (double *) malloc(tamVetor    * sizeof(double));
	h_vetorB = (double *) malloc(tamVetor    * sizeof(double));
	h_vetorC = (double *) malloc(numBlocks.x * sizeof(double));

	//Inicializando o vetor no host com valores aleatórios
	for(int i = 0; i < tamVetor; i++){
		h_vetorA[i] = (double)rand()/(double)RAND_MAX;
		h_vetorB[i] = (double)rand()/(double)RAND_MAX;
		//h_vetorA[i] = 1.0;
	}

//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO HOST

	h_inicio = clock();
	//Soma todas as posições do vetor no host
	for(int i = 0; i < tamVetor; i++){
		h_resultado += h_vetorA[i] * h_vetorB[i];
	}
	h_fim = clock();

//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO DEVICE

	cudaEventRecord(d_inicioTotal);

	//Aloca vetor no device
	cudaMalloc(&d_vetorA, tamVetor    * sizeof(double)); cudaTest("Falha ao alocar d_vetorA");
	cudaMalloc(&d_vetorB, tamVetor    * sizeof(double)); cudaTest("Falha ao alocar d_vetorB");
	cudaMalloc(&d_vetorC, numBlocks.x * sizeof(double)); cudaTest("Falha ao alocar d_vetorC");

	//Transferindo valores para o device
	cudaMemcpy(d_vetorA, h_vetorA, tamVetor * sizeof(double), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_vetorA para d_vetorA");
	cudaMemcpy(d_vetorB, h_vetorB, tamVetor * sizeof(double), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_vetorB para d_vetorB");

	cudaEventRecord(d_inicioKernel);
	//Primeiro multiplica cada elemento dos vetores
	multiplyElements<<<numBlocks, numThreads>>>(tamVetor, d_vetorA, d_vetorB);
	cudaTest("Erro na execução do Kernel multiplyElements");

	//Depois soma todos os valores
	vectorSum<<<numBlocks, threadsPerBlock>>>(d_vetorA, d_vetorC);
	cudaTest("Erro na execução do Kernel vectorSum");

	dim3 numBlocks2(numBlocks.x/threadsPerBlock.x);

	//Se ainda tiverem blocos suficientes, executa uma segunda vez
	if(numBlocks2.x > 0){
		
		vectorSum<<<numBlocks2, threadsPerBlock>>>(d_vetorC, d_vetorA);
		cudaEventRecord(d_fimKernel);
		cudaTest("Erro ao executar o Kernel");

		//Copia o vetor de volta para o host e soma o restante
		cudaMemcpy(h_vetorC, d_vetorA, numBlocks2.x * sizeof(double), cudaMemcpyDeviceToHost);
		cudaTest("Erro ao copiar d_vetorA para h_vetorB");

		for(int i = 0; i < numBlocks2.x; i++){
			d_resultado += h_vetorC[i];
		}

	//Senão apenas copia de volta para o host e soma o restante
	}else{

		cudaEventRecord(d_fimKernel);

		cudaMemcpy(h_vetorC, d_vetorC, numBlocks.x * sizeof(double), cudaMemcpyDeviceToHost);
		cudaTest("Erro ao copiar d_vetorA para h_vetorB");

		for(int i = 0; i < numBlocks.x; i++){
			d_resultado += h_vetorC[i];
		}

	}

	cudaEventRecord(d_fimTotal);

//------------------------------------------------------------------------------------------------------	
//EXIBIÇÃO DOS RESULTADOS

	printf("Resultado CPU: %lf\n", h_resultado);
	printf("Resultado GPU: %lf\n", d_resultado);
	printf("------------------------------------\n");
//------------------------------------------------------------------------------------------------------	
//CÁLCULO DE TEMPO GASTO NAS EXECUÇÕES

	//Calculo do tempo gasto no host
	double tempo_host = ((double) (h_fim - h_inicio)) / CLOCKS_PER_SEC;
	printf("Tempo CPU: %fs\n", tempo_host);

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

	free(h_vetorA);
	free(h_vetorB);
	free(h_vetorC);

	cudaFree(d_vetorA);
	cudaFree(d_vetorB);
	cudaFree(d_vetorC);

	cudaEventDestroy(d_inicioTotal);
	cudaEventDestroy(d_fimTotal);
	cudaEventDestroy(d_inicioKernel);
	cudaEventDestroy(d_fimKernel);

	cudaDeviceReset();

	return 0;
}
