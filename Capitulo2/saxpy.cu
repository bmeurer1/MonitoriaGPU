#include <stdio.h>
#include <cuda.h>

#define NUMTHREADS 512

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

__global__ void saxpy(float *vetorA, float *vetorB, float *vetorC, float scalar, float tamVetor){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < tamVetor){
		vetorC[tid] = vetorA[tid] * scalar + vetorB[tid];
	}
}

int main(int argc, char const *argv[]){

	float tamVetor = 0;
	int numThreads = 0;
	float 	scalar = 0;

	if(argc != 4){
		printf("Modo de uso: %s <Tamanho do Vetor> <Scalar> <Threads Por Bloco>\n", argv[0]);
		exit(-1);
	}else{
		tamVetor 	= atoi(argv[1]);
		scalar 		= atoi(argv[2]);
		numThreads  = atoi(argv[3]);
 
		if(numThreads > 1024 || numThreads < 1){
			printf("Threads Por Bloco tem que ser entre 1 e 1024\n");
		}
	}

//------------------------------------------------------------------------------------------------------	
//DECLARAÇÃO DE VARIÁVEIS

	float *h_vetorA, *h_vetorB, *h_vetorC, *h_resultadoGPU;
	float *d_vetorA, *d_vetorB, *d_vetorC;

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

	//Aloca os vetores no host
	h_vetorA  	   = (float *) malloc(tamVetor * sizeof(float));
	h_vetorB 	   = (float *) malloc(tamVetor * sizeof(float));
	h_vetorC 	   = (float *) malloc(tamVetor * sizeof(float));
	h_resultadoGPU = (float *) malloc(tamVetor * sizeof(float));

	//Inicializa os vetores no host
	for(int i = 0; i < tamVetor; i++){
		h_vetorA[i] = (double)rand()/(double)RAND_MAX;
		h_vetorB[i] = (double)rand()/(double)RAND_MAX;
	}

//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO DEVICE

	cudaEventRecord(d_inicioTotal);
	
	//Aloca os vetores no device
	cudaMalloc(&d_vetorA, tamVetor * sizeof(float)); cudaTest("Falha ao alocar memória para o vetor d_vetorA");
	cudaMalloc(&d_vetorB, tamVetor * sizeof(float)); cudaTest("Falha ao alocar memória para o vetor d_vetorB");
	cudaMalloc(&d_vetorC, tamVetor * sizeof(float)); cudaTest("Falha ao alocar memória para o vetor d_vetorC");

	//Transferindo valores para o device
	cudaMemcpy(d_vetorA, h_vetorA, tamVetor * sizeof(float), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_vetorA para d_vetorA");
	cudaMemcpy(d_vetorB, h_vetorB, tamVetor * sizeof(float), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_vetorB para d_vetorB");

	//Determina quantidade de threads e blocos a serem lançados no kernel
	dim3 threadsPerBlock(numThreads);
	dim3 numBlocks((tamVetor + threadsPerBlock.x - 1)/threadsPerBlock.x);

	//Executa o saxpy no device
	cudaEventRecord(d_inicioKernel);
	saxpy<<<numBlocks, threadsPerBlock>>>(d_vetorA, d_vetorB, d_vetorC, scalar, tamVetor);
	cudaTest("Erro ao executar o Kernel");
	cudaEventRecord(d_fimKernel);
	

	//IMPORTANTE
	//Kernels em CUDA aceitam tanto o tipo dim3 quanto o tipo int
	//para as quantidades de threads e blocos. O código abaixo
	//abaixo funcionará exatamente da mesma maneira do que o que
	//foi escrito acima. Faça um teste para comprovar :D

	/*
	//Determina quantidade de threads e blocos a serem lançados no kernel
	int threadsPerBlock = numThreads;
	int numBlocks = ((tamVetor + threadsPerBlock - 1)/threadsPerBlock);
	
	//Executa o saxpy no device
	cudaEventRecord(d_inicioKernel);
	saxpy<<<numBlocks, threadsPerBlock>>>(d_vetorA, d_vetorB, d_vetorC, scalar, tamVetor);
	cudaTest("Erro ao executar o Kernel");
	cudaEventRecord(d_fimKernel);
	*/

	//Traz os resultados de volta para o host
	cudaMemcpy(h_resultadoGPU, d_vetorC, tamVetor * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(d_fimTotal);
	cudaTest("Erro ao copiar resultado para o vetor h_resultadoGPU");


//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO HOST

	h_inicio = clock();
	//Executa o saxpy no host
	for(int i = 0; i < tamVetor; i++){
		h_vetorC[i] = h_vetorA[i] * scalar + h_vetorB[i];
	}
	h_fim = clock();

//------------------------------------------------------------------------------------------------------	
//COMPARAÇÃO DE RESULTADOS

	//Compara vetores para ver se os cálculos foram corretos
	int erro = 0;
	for(int i = 0; i < tamVetor; i++){
		if(fabs(h_vetorC[i] - h_resultadoGPU[i]) > 0.00001){
			erro = 1;
		}
	}

	if(!erro){
		printf("Resultado Correto\n");
	}else{
		printf("Resultado Incorreto\n");
	}
	printf("------------------------------------\n");

//------------------------------------------------------------------------------------------------------	
//CÁLCULO DE TEMPO GASTO NAS EXECUÇÕES

	//Calculo do tempo gasto no host
	float tempo_host = ((float) (h_fim - h_inicio)) / CLOCKS_PER_SEC;
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
	free(h_resultadoGPU);

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