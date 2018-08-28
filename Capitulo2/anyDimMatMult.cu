#include <stdio.h>
#include <cuda.h>


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

__global__ void matMul(float *matrizA, float *matrizB, float *matrizC, int linhasA, int linhasB, int colunasA, int colunasB){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0;
	for(int k = 0; k < colunasA; k++){
		sum += matrizA[i * colunasA + k] * matrizB[k * colunasB + j];
	}

	matrizC[i * colunasB + j] = sum;

}

int main(int argc, char const *argv[]){

	int real_linhasA;
	int real_linhasB;
	int real_colunasA;
	int real_colunasB;
	int block_size;

	if(argc != 6){
		printf("Modo de uso: %s <Linhas Matriz A> <Colunas Matriz A> <Linhas Matriz B> <Colunas Matriz B> <Block Size>\n", argv[0]);
		exit(-1);
	}else{
		real_linhasA  = atoi(argv[1]);
		real_colunasA = atoi(argv[2]);
		real_linhasB  = atoi(argv[3]);
		real_colunasB = atoi(argv[4]);
		block_size 	  = atoi(argv[5]);

		if(real_colunasA != real_linhasB){
			printf("Erro: Colunas A != Linhas B\n");
			exit(-1);
		}

		if(block_size > 32 || block_size < 1){
			printf("Erro: Block Size tem que ser entre 1 e 32\n");
			exit(-1);
		}
	}

	int linhasA;
	int linhasB;
	int colunasA;
	int colunasB;

	//Completa as linhas e colunas para serem múltiplas de block_size
	linhasA  = real_linhasA  + (real_linhasA  % block_size);
	colunasA = real_colunasA + (real_colunasA % block_size);

	linhasB  = real_linhasB  + (real_linhasB  % block_size);
	colunasB = real_colunasB + (real_colunasB % block_size);

//------------------------------------------------------------------------------------------------------	
//DECLARAÇÃO DE VARIÁVEIS

	float *h_matrizA, *h_matrizB, *h_matrizC, *h_matrizGPU;
	float *d_matrizA, *d_matrizB, *d_matrizC;

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

	//Aloca os vetores no host
	h_matrizA 	= (float *) malloc(linhasA * colunasA * sizeof(float));
	h_matrizB 	= (float *) malloc(linhasB * colunasB * sizeof(float));
	h_matrizC 	= (float *) malloc(linhasA * colunasB * sizeof(float));
	h_matrizGPU = (float *) malloc(linhasA * colunasB * sizeof(float));

	//Inicializa os vetores no host
	for(int i = 0; i < linhasA * colunasA; i++){

		if(i < real_linhasA || i < real_colunasA){
			h_matrizA[i] = (double)rand()/(double)RAND_MAX;
		}else{
			//Preenche com zeros o que completamos anteriormente
			h_matrizA[i] = 0;
		}	
	}

	//Inicializa os vetores no host
	for(int i = 0; i < linhasB * colunasB; i++){
		if(i < real_linhasB || i < real_colunasB){
			h_matrizB[i] = (double)rand()/(double)RAND_MAX;
		}else{
			//Preenche com zeros o que completamos anteriormente
			h_matrizB[i] = 0;
		}	
	}


//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO DEVICE

	cudaEventRecord(d_inicioTotal);

	//Aloca as matrizes no device
	cudaMalloc(&d_matrizA, linhasA * colunasA * sizeof(float)); cudaTest("Falha ao alocar memória para a matriz d_matrizA");
	cudaMalloc(&d_matrizB, linhasB * colunasB * sizeof(float)); cudaTest("Falha ao alocar memória para a matriz d_matrizB");
	cudaMalloc(&d_matrizC, linhasA * colunasB * sizeof(float)); cudaTest("Falha ao alocar memória para a matriz d_matrizC");

	//Transferindo valores para o device
	cudaMemcpy(d_matrizA, h_matrizA, linhasA * colunasA * sizeof(float), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_matrizA para d_matrizA");
	cudaMemcpy(d_matrizB, h_matrizB, linhasB * colunasB * sizeof(float), cudaMemcpyHostToDevice); cudaTest("Falha ao transferir de h_matrizA para d_matrizB");


	//Determina quantidade de threads e blocos a serem lançados no kernel
	dim3 threadsPerBlock(block_size, block_size);
	dim3 numBlocks(linhasA/threadsPerBlock.x, colunasB/threadsPerBlock.y);

	//Multiplica as matrizes no device
	cudaEventRecord(d_inicioKernel);
	matMul<<<numBlocks, threadsPerBlock>>>(d_matrizA, d_matrizB, d_matrizC, linhasA, linhasB, colunasA, colunasB);
	cudaTest("Erro ao executar o Kernel");
	cudaEventRecord(d_fimKernel);

	//Traz os resultados de volta para o host
	cudaMemcpy(h_matrizGPU, d_matrizC, linhasA * colunasB * sizeof(float), cudaMemcpyDeviceToHost);
	cudaTest("Erro ao copiar resultado para o vetor h_resultadoGPU");

	cudaEventRecord(d_fimTotal);

//------------------------------------------------------------------------------------------------------	
//EXECUÇÃO NO HOST

	h_inicio = clock();
	//Multiplica as matrizes no host
	for(int i = 0; i < linhasA; i++){
		for(int j = 0; j < colunasB; j++){
			float sum = 0;
			for(int k = 0; k < colunasA; k++){
				sum += h_matrizA[i * colunasA + k] * h_matrizB[k * colunasB + j];
			}
			h_matrizC[i * colunasB + j] = sum;
		}	
	}
	h_fim = clock();


//------------------------------------------------------------------------------------------------------	
//COMPARAÇÃO DE RESULTADOS

	//Compara vetores para ver se os cálculos foram corretos
	int erro = 0;
	for(int i = 0; i < linhasA * colunasB; i++){
		if(fabs(h_matrizC[i] - h_matrizGPU[i]) > 0.00001){
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

	free(h_matrizA);
	free(h_matrizB);
	free(h_matrizC);
	free(h_matrizGPU);

	cudaFree(d_matrizA);
	cudaFree(d_matrizB);
	cudaFree(d_matrizC);

	cudaEventDestroy(d_inicioTotal);
	cudaEventDestroy(d_fimTotal);
	cudaEventDestroy(d_inicioKernel);
	cudaEventDestroy(d_fimKernel);

	cudaDeviceReset();
	
	return 0;
}