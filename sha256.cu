#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdint.h>

// check cudaMalloc error
#define CHECK(call)                                            \
{                                                              \
    const cudaError_t error = call;                            \
    if (error != cudaSuccess)                                  \
    {                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error,       \
                cudaGetErrorString(error));                    \
        exit(1);                                               \
    }                                                          \
}

// logic functions
#define ROTL(W, n) (((W << n) & 0xFFFFFFFF) | (W) >> (32 - (n)))
#define SHR(W, n) ((W >> n) & 0xFFFFFFFF)
#define Conditional(x, y, z) ((x & y) ^ ((~x) & z))
#define Majority(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define LSigma_0(x) (ROTL(x, 30) ^ ROTL(x, 19) ^ ROTL(x, 10))
#define LSigma_1(x) (ROTL(x, 26) ^ ROTL(x, 21) ^ ROTL(x, 7))
#define SSigma_0(x) (ROTL(x, 25) ^ ROTL(x, 14) ^ SHR(x, 3))
#define SSigma_1(x) (ROTL(x, 15) ^ ROTL(x, 13) ^ SHR(x, 10))

// the path of file
const char *FILEPATH = "/home/chenq/cuda/0.txt";

// the numbers of characters per reading file 600LLU * 1024 * 1024
uint64_t READSIZE_MAX = 700LLU * 1024 * 1024;

// the size of a data block per layer
uint64_t DATABLOCKSIZE[2] = {0LLU, 0LLU};

// the number of characters for padding per layer
uint64_t PADDINGSIZE[2] = {0LLU, 0LLU};

// the number of data block amount
uint64_t DATABLOCKAMOUNT[2] = {0LLU, 0LLU};

// the dimension of block
uint64_t BLOCKDIMENSION_1[3]= {32llu, 1llu, 1llu};
uint64_t BLOCKDIMENSION_2[3]= {128llu, 1llu, 1llu};

// the dimension of grid
uint64_t GRIDDIMENSION_1[3]= {1llu, 1llu, 1llu};
uint64_t GRIDDIMENSION_2[3]= {1llu, 1llu, 1llu};

// recording time in seconds
double getTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// preprocess
void preprocess(const uint64_t charactersPerReading);

// set up thread configuration
void threadConfig(uint64_t threadamount, uint64_t *blockDimension, uint64_t *gridDimension);

// padding characters
__global__ void paddingChar(unsigned char *D_C, unsigned char *D_P, uint64_t DATABLOCKSIZE0, uint64_t DATABLOCKSIZE1, uint64_t PADDINGSIZE0, uint64_t PADDINGSIZE1, uint64_t dataBlockAmount);

// transform 4 unsigned char to 1 32-bit unsigned int
__global__ void unsignedCharToUnsignedInt(const unsigned char *D_P, uint32_t *D_T, uint64_t dataBlockAmount, uint64_t groupsPerThread_1, uint64_t groupsPerThread_2);

// extending 16 32-bit integers to 64 32-bit integers
__global__ void extending(uint32_t *D_T, uint32_t *D_E, uint64_t dataBlockAmount, uint64_t groupsPerThread_1, uint64_t groupsPerThread_2);

// updating hash value
__global__ void updatingHashValue(const uint32_t *D_E, uint32_t *D_H, uint64_t DATABLOCKSIZE0, uint64_t DATABLOCKSIZE1, uint64_t PADDINGSIZE0, uint64_t PADDINGSIZE1, uint64_t dataBlockAmount, uint64_t layer, bool oddDataBlockAmount, uint64_t hashValuePosition);

// little end to big end
__global__ void lend_to_bend(uint32_t *V, uint64_t threads);

// main function
int main(int agrc, char *argv[])
{
    printf("\nComputing hash value on GPU_CUDA.\n");

    // input data block size coefficient
    uint64_t coef = 0;
    printf("Please enter the coefficient of the size of the data block in KB: ");
    scanf("%llu", &coef);

    // set the start time
    double start, end, phase_1;
    start = getTime();

    // convert argv[] to int
    int argv_2 = atoi(argv[2]);
    int argv_3 = atoi(argv[3]);
    // int argv_4 = atoi(argv[4]);

    // get the file size
    printf("Have read file: %s\n", argv[1]);
    FILE *fin;
    fin = fopen(argv[1], "rb");
    if (!fin)
    {
        printf("Reading file failed.\n");
        if (agrc == 1)
            printf("Please enter file name.\n");
        exit(EXIT_FAILURE);
    }
    fseek(fin, 0, SEEK_END);
    uint64_t fileSize = ftell(fin);
    rewind(fin);
    printf("The size of file: %llu Bytes\n", fileSize);

    // determine the size of data block
    if (coef > 0)
    {
        DATABLOCKSIZE[0] = coef * 1024;
        if (DATABLOCKSIZE[0] > fileSize || DATABLOCKSIZE[0] > READSIZE_MAX)
        {
            printf("Data block is too big.");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        DATABLOCKSIZE[0] = fileSize;
    }

    // get the number of characters per reading and the reading times
    uint64_t charactersPerReading = 0;
    if (fileSize <= READSIZE_MAX + 20 * 1024 * 1024)
    {
        charactersPerReading = fileSize;
    }
    else
    {
        charactersPerReading = (READSIZE_MAX / DATABLOCKSIZE[0]) * DATABLOCKSIZE[0];
    }
    uint64_t readTimes = fileSize / charactersPerReading;
    if (fileSize % charactersPerReading > 0) 
        readTimes++;

    // get the number of layers in the Merkle Hash Tree
    uint64_t layers = 1;
    uint64_t layerProcess = fileSize / DATABLOCKSIZE[0];
    if (fileSize % DATABLOCKSIZE[0] > 0)
        layerProcess++;
    while (layerProcess != 1)
    {
        if (layerProcess % 2 != 0)
            layerProcess++;
        layerProcess = layerProcess / 2;
        layers++;
    }

// *********************************************************************************
// ****************** Computing 0 layer hash value *********************************
// *********************************************************************************

    // get the number of data block
    uint64_t dataBlockAmount = fileSize / DATABLOCKSIZE[0];
    if (fileSize % DATABLOCKSIZE[0] > 0)
        dataBlockAmount++;

    // determining the parity of data block amount
    bool oddDataBlockAmount = false;
    if (dataBlockAmount % 2 != 0)
        oddDataBlockAmount = true;

    // get the number of hash value
    uint64_t hashValueAmount = dataBlockAmount;
    if ((hashValueAmount % 2 != 0) && layers > 1)
        hashValueAmount++;
    uint64_t hashValueAmountArray[layers];
    hashValueAmountArray[0] = hashValueAmount;  

    // get data block size, padding characters, data block amount (per reading) and storage size (per reading)
    uint64_t dataBlockAmountPerReading = 0;
    uint64_t storageSizePerReading = 0;
    preprocess(charactersPerReading);
    printf("data_block_0 size = %lu  data_block_1 size = %lu\n", DATABLOCKSIZE[0], DATABLOCKSIZE[1]);
    printf("padding_block_0 size = %lu  padding_block_1 size = %lu\n", PADDINGSIZE[0], PADDINGSIZE[1]);
    printf("data_block_0 amount = %lu  data_block_1 amount = %lu\n", DATABLOCKAMOUNT[0], DATABLOCKAMOUNT[1]);

    // get dataBlockAmountPerReading and storageSizePerReading
    dataBlockAmountPerReading = DATABLOCKAMOUNT[0] + DATABLOCKAMOUNT[1];
    storageSizePerReading = (DATABLOCKSIZE[0] + PADDINGSIZE[0]) * DATABLOCKAMOUNT[0] + (DATABLOCKSIZE[1] + PADDINGSIZE[1]) * DATABLOCKAMOUNT[1];
    printf("data block amount: %lu\n", dataBlockAmountPerReading);
    printf("storageSizePerReading = %lu\n", storageSizePerReading);

    // set up thread config 1
    uint64_t blockDimension_1[3] = {argv_2, 1llu, 1llu};
    uint64_t gridDimension_1[3] = {1llu, 1llu, 1llu};
    threadConfig(dataBlockAmountPerReading, blockDimension_1, gridDimension_1);
    dim3 block_1(blockDimension_1[0], blockDimension_1[1], blockDimension_1[2]);
    dim3 grid_1(gridDimension_1[0], gridDimension_1[1], gridDimension_1[2]);
    printf("block_dimension_1: %lu, %lu, %lu\n", blockDimension_1[0], blockDimension_1[1], blockDimension_1[2]);
    printf("grid_dimension_1: %lu, %lu, %lu\n",  gridDimension_1[0],  gridDimension_1[1],  gridDimension_1[2]);

    // set up thread config 2
    uint64_t blockDimension_2[3] = {argv_3, 1llu, 1llu};
    uint64_t gridDimension_2[3] = {1llu, 1llu, 1llu};
    uint64_t groupsPerThread_1 = 1llu;
    uint64_t groupsPerThread_2 = 0llu;
    uint64_t groupsTotal = storageSizePerReading / 64;
    uint64_t threads = 0;
    const uint64_t blocksInGrid = 28;
    
    groupsPerThread_1 = groupsTotal / (argv_3 * blocksInGrid);
    if (groupsTotal % (argv_3 * blocksInGrid) > 0) ++groupsPerThread_1; 

     
    if  (storageSizePerReading >= 64 * groupsPerThread_1) 
    {
        groupsPerThread_2 = (storageSizePerReading % (64 * groupsPerThread_1)) / 64;
    }
    else
    {
        printf("Don't correctly set up thread config 2\n ");
        exit(-1);
    }

    threads = groupsTotal / groupsPerThread_1;
    if (groupsPerThread_2 > 0) ++threads;

    threadConfig(threads, blockDimension_2, gridDimension_2);
    dim3 block_2(blockDimension_2[0], blockDimension_2[1], blockDimension_2[2]);
    dim3 grid_2(gridDimension_2[0], gridDimension_2[1], gridDimension_2[2]);

    printf("group_1 = %llu, group_2 = %llu\n", groupsPerThread_1, groupsPerThread_2);
    printf("block dimension 2: %lu, %lu, %lu\n", blockDimension_2[0], blockDimension_2[1], blockDimension_2[2]);
    printf("grid dimension 2: %lu, %lu, %lu\n", gridDimension_2[0], gridDimension_2[1], gridDimension_2[2]);

    // data stream
    char *C = NULL;
    char *D_C = NULL;

    // storing the data after padding
    unsigned char *D_P = NULL;

    //  storing the data after transform
    uint32_t *D_T = NULL;

    // storing the data after extending
    uint32_t *D_E = NULL;

    // assign the storage space of hash value
    uint32_t *D_V[layers];
    CHECK(cudaMalloc((uint32_t **)&D_V[0], hashValueAmountArray[0] * 8 * sizeof(uint32_t)));

    // hash value position using in computation of 0 layer
    uint64_t hashValuePosition = 0;

    // parallelly updating data block's hash value
    for (uint64_t i = 0; i < readTimes; ++i)
    {
        // determining data block amount and storage size for last reading
        if (readTimes > 1 && i == readTimes - 1 && fileSize % charactersPerReading != 0)
        {
            // last time reading data
            charactersPerReading = fileSize % charactersPerReading;
            preprocess(charactersPerReading);
            dataBlockAmountPerReading = DATABLOCKAMOUNT[0] + DATABLOCKAMOUNT[1];
            storageSizePerReading = (DATABLOCKSIZE[0] + PADDINGSIZE[0]) * DATABLOCKAMOUNT[0] + (DATABLOCKSIZE[1] + PADDINGSIZE[1]) * DATABLOCKAMOUNT[1];

            // set up thread config 1
            uint64_t blockDimension_1[3] = {argv_2, 1llu, 1llu};
            uint64_t gridDimension_1[3] = {1llu, 1llu, 1llu};
            threadConfig(dataBlockAmountPerReading, blockDimension_1, gridDimension_1);
            dim3 block_1(blockDimension_1[0], blockDimension_1[1], blockDimension_1[2]);
            dim3 grid_1(gridDimension_1[0], gridDimension_1[1], gridDimension_1[2]);
            printf("block_dimension_1: %lu, %lu, %lu\n", blockDimension_1[0], blockDimension_1[1], blockDimension_1[2]);
            printf("grid_dimension_1: %lu, %lu, %lu\n",  gridDimension_1[0],  gridDimension_1[1],  gridDimension_1[2]);

            // set up thread config 2
            uint64_t blockDimension_2[3] = {argv_3, 1llu, 1llu};
            uint64_t gridDimension_2[3] = {1llu, 1llu, 1llu};
            uint64_t groupsPerThread_1 = 1llu;
            uint64_t groupsPerThread_2 = 0llu;
            uint64_t groupsTotal = storageSizePerReading / 64;
            groupsPerThread_1 = groupsTotal / (argv_3 * blocksInGrid);
            if (groupsTotal % (argv_3 * blocksInGrid) > 0) 
                ++groupsPerThread_1; 

            uint64_t threads = groupsTotal / groupsPerThread_1; 
            if  (storageSizePerReading >= 64 * groupsPerThread_1) 
            {
                groupsPerThread_2 = (storageSizePerReading % (64 * groupsPerThread_1)) / 64;
            }
            else
            {
                printf("Don't correctly set up thread config 2\n ");
                exit(-1);
            }
            if (groupsPerThread_2 > 0) ++threads;
            threadConfig(threads, blockDimension_2, gridDimension_2);
            dim3 block_2(blockDimension_2[0], blockDimension_2[1], blockDimension_2[2]);
            dim3 grid_2(gridDimension_2[0], gridDimension_2[1], gridDimension_2[2]);
            printf("group_1 = %llu, group_2 = %llu\n", groupsPerThread_1, groupsPerThread_2);
            printf("block dimension 2: %lu, %lu, %lu\n", blockDimension_2[0], blockDimension_2[1], blockDimension_2[2]);
            printf("grid dimension 2: %lu, %lu, %lu\n", gridDimension_2[0], gridDimension_2[1], gridDimension_2[2]);
        }

        // read characters from input data stream and transfer data from host to device
        C = (char *)malloc(charactersPerReading);
        CHECK(cudaMalloc((char **)&D_C, charactersPerReading));
        fread(C, charactersPerReading, 1, fin);
        cudaMemcpy(D_C, C, charactersPerReading, cudaMemcpyHostToDevice);
        free(C);

        // padding characters
        CHECK(cudaMalloc((unsigned char **)&D_P, storageSizePerReading));
        paddingChar<<<grid_1, block_1>>>((unsigned char *)D_C, D_P, DATABLOCKSIZE[0], DATABLOCKSIZE[1], PADDINGSIZE[0], PADDINGSIZE[1], dataBlockAmountPerReading);
        cudaDeviceSynchronize();
        cudaFree(D_C);

        // transform 4 unsigned char to 1 32-bit unsigned int
        CHECK(cudaMalloc((uint32_t **)&D_T, storageSizePerReading));
        unsignedCharToUnsignedInt<<<grid_2, block_2>>>(D_P, D_T, threads, groupsPerThread_1, groupsPerThread_2);
        cudaDeviceSynchronize();
        cudaFree(D_P);

        // extending 16 32-bit integers to 64 32-bit integers
        CHECK(cudaMalloc((uint32_t **)&D_E, 4 * storageSizePerReading));
        extending<<<grid_2, block_2>>>(D_T, D_E, threads, groupsPerThread_1, groupsPerThread_2);
        cudaDeviceSynchronize();
        cudaFree(D_T);

        // updating hash value
        updatingHashValue<<<grid_1, block_1>>>(D_E, D_V[0], DATABLOCKSIZE[0], DATABLOCKSIZE[1], PADDINGSIZE[0], PADDINGSIZE[1], dataBlockAmountPerReading, 0LLU, (oddDataBlockAmount && (i == readTimes - 1)), hashValuePosition);
        cudaDeviceSynchronize();
        cudaFree(D_E);
        hashValuePosition += (dataBlockAmountPerReading * 8);

        // check kernel lauch fail
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            exit(-1);
        }
    }

    // transform little end to big end
    uint64_t blockDimension_3[3] = {32, 1, 1};
    uint64_t gridDimension_3[3] = {1, 1, 1};
    threadConfig(hashValueAmountArray[0] * 8, blockDimension_3, gridDimension_3);
    dim3 block_3(blockDimension_3[0], blockDimension_3[1], blockDimension_3[2]);
    dim3 grid_3(gridDimension_3[0], gridDimension_3[1], gridDimension_3[2]);
    lend_to_bend<<<grid_3, block_3>>>(D_V[0], hashValueAmountArray[0] * 8);

    // recording phase 1 time
    phase_1 = getTime();

// *********************************************************************************
// ****************** Computing 2 ~ (layers - 1) layer hash value ******************
// *********************************************************************************

    // preprocess
    DATABLOCKSIZE[0] = 64LLU;
    DATABLOCKSIZE[1] = 0;
    PADDINGSIZE[0] = 64LLU;
    PADDINGSIZE[1] = 0;

    // computing hash value for 1 to (layers-1) layer
    for (uint64_t l = 1; l < layers; l++)
    {
        // updating the number of data block
        uint64_t dataBlockAmount = hashValueAmountArray[l - 1] / 2;

        // updating storage size
        uint64_t storageSize = (DATABLOCKSIZE[0] + PADDINGSIZE[0]) * dataBlockAmount;

        // updating the parity of data block amount
        oddDataBlockAmount = false;
        if (dataBlockAmount % 2 != 0)
            oddDataBlockAmount = true;

        // updating the number of hash value
        hashValueAmount = dataBlockAmount;
        if (oddDataBlockAmount && l != layers - 1)
            hashValueAmount++;
        hashValueAmountArray[l] = hashValueAmount;

        // set up thread config 1
        uint64_t blockDimension_1[3] = {32llu, 1llu, 1llu};
        uint64_t gridDimension_1[3] = {1llu, 1llu, 1llu};
        threadConfig(dataBlockAmount, blockDimension_1, gridDimension_1);
        dim3 block_1(blockDimension_1[0], blockDimension_1[1], blockDimension_1[2]);
        dim3 grid_1(gridDimension_1[0], gridDimension_1[1], gridDimension_1[2]);

        // set up thread config 2
        uint64_t blockDimension_2[3] = {128llu, 1llu, 1llu};
        uint64_t gridDimension_2[3] = {1llu, 1llu, 1llu};
        threadConfig(storageSizePerReading / 64, blockDimension_2, gridDimension_2);
        dim3 block_2(blockDimension_2[0], blockDimension_2[1], blockDimension_2[2]);
        dim3 grid_2(gridDimension_2[0], gridDimension_2[1], gridDimension_2[2]);

        // get data from the previous hash value
        cudaMalloc((char **)&D_C, hashValueAmountArray[l - 1] * 8 * sizeof(uint32_t));
        cudaMemcpy(D_C, D_V[l - 1], hashValueAmountArray[l - 1] * 8 * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

        // padding characters
        CHECK(cudaMalloc((char **)&D_P, storageSize));
        paddingChar<<<grid_1, block_1>>>((unsigned char *)D_C, D_P, DATABLOCKSIZE[0], DATABLOCKSIZE[1], PADDINGSIZE[0], PADDINGSIZE[1], dataBlockAmount);
        cudaDeviceSynchronize();
        cudaFree(D_C);

        // transform 4 unsigned char to 1 32-bit unsigned int
        CHECK(cudaMalloc((char **)&D_T, storageSize));
        unsignedCharToUnsignedInt<<<grid_2, block_2>>>(D_P, D_T, storageSize / 64, 1, 1);
        cudaDeviceSynchronize();
        cudaFree(D_P);

        // extending 16 32-bit integers to 64 32-bit integers
        CHECK(cudaMalloc((char **)&D_E, 4 * storageSize));
        extending<<<grid_2, block_2>>>(D_T, D_E, storageSize /64, 1, 1);
        cudaDeviceSynchronize();
        cudaFree(D_T);

        // updating hash value
        CHECK(cudaMalloc((uint32_t **)&D_V[l], hashValueAmount * 8 * sizeof(uint32_t)));
        updatingHashValue<<<grid_1, block_1>>>(D_E, D_V[l], DATABLOCKSIZE[0], DATABLOCKSIZE[1], PADDINGSIZE[0], PADDINGSIZE[1], dataBlockAmount, l, oddDataBlockAmount, 0llu);
        cudaDeviceSynchronize();
        cudaFree(D_E);

        // transform little end to big end
        uint64_t blockDimension_3[3] = {32, 1, 1};
        uint64_t gridDimension_3[3] = {1, 1, 1};
        threadConfig(hashValueAmountArray[l] * 8, blockDimension_3, gridDimension_3);
        dim3 block_3(blockDimension_3[0], blockDimension_3[1], blockDimension_3[2]);
        dim3 grid_3(gridDimension_3[0], gridDimension_3[1], gridDimension_3[2]);
        lend_to_bend<<<grid_3, block_3>>>(D_V[l], hashValueAmountArray[l] * 8);
    }

    // assign the storage space of the hash value for per layer on host side
    uint32_t *V[layers];
    for (uint32_t i = 0; i < layers; i++)
    {
        V[i] = (uint32_t *)malloc(hashValueAmountArray[i] * 8 * sizeof(uint32_t));
    }

    // transfer hash value from device to host
    for (uint32_t i = 0; i < layers; i++)
    {
        cudaMemcpy(V[i], D_V[i], hashValueAmountArray[i] * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    // set the end time
    end = getTime();

    // print the merkle root
    printf("The Merkle Root: ");
    unsigned char *Temp = (unsigned char *)&V[layers - 1][0];
    for (uint64_t j = 0; j < 32; ++j)
    {
        printf("%02x", Temp[j]);
    }
    printf("\n");

    // free data pointer
    fclose(fin);
    for (uint64_t i = 0; i < layers; i++)
    {
        free(V[i]);
    }
    for (uint64_t i = 0; i < layers; i++)
    {
        cudaFree(D_V[i]);
    }

    // print time consumption
    printf("%f\n\n", end - start);
    
    return 0;
}

// get data block size, padding characters, data block amount and storage size
void preprocess(const uint64_t charactersPerReading)
{
    // 1. get the the size of data block for per reading
    DATABLOCKSIZE[1] = 0;
    if (charactersPerReading % DATABLOCKSIZE[0] > 0)
        DATABLOCKSIZE[1] = charactersPerReading % DATABLOCKSIZE[0];

    // 2. get the number of characters of padding for per reading
    if (DATABLOCKSIZE[0] % 64 < 56)
    {
        PADDINGSIZE[0] = 56 - (DATABLOCKSIZE[0] % 64) + 8;
    }
    else
    {
        PADDINGSIZE[0] = 64 - (DATABLOCKSIZE[0] % 64) + 56 + 8;
    }
    if (DATABLOCKSIZE[1] > 0)
    {
        if (DATABLOCKSIZE[1] % 64 < 56)
        {
            PADDINGSIZE[1] = 56 - (DATABLOCKSIZE[1] % 64) + 8;
        }
        else
        {
            PADDINGSIZE[1] = 64 - (DATABLOCKSIZE[1] % 64) + 56 + 8;
        }
    }

    // 3. get the number of data block for per reading
    DATABLOCKAMOUNT[0] = charactersPerReading / DATABLOCKSIZE[0];
    DATABLOCKAMOUNT[1] = 0;
    if (DATABLOCKSIZE[1] > 0)
        DATABLOCKAMOUNT[1] = 1;
} 

// set up thread configuration
void threadConfig(uint64_t threadamount, uint64_t *blockDimension, uint64_t *gridDimension) 
{
    // set up block and gird dimension 1
    if (threadamount > blockDimension[0] * blockDimension[1] * blockDimension[2])
    {
        gridDimension[0] = threadamount / (blockDimension[0] * blockDimension[1] * blockDimension[2]);
        if (threadamount % blockDimension[0] > 0)
            gridDimension[0]++;
        gridDimension[1] = 1;
        gridDimension[2] = 1;
    }
    else
    {
        gridDimension[0] = 1;
        gridDimension[1] = 1;
        gridDimension[2] = 1;
    }
}

// padding characters, data from D_C to D_P
__global__ void paddingChar(unsigned char *D_C, unsigned char *D_P, uint64_t DATABLOCKSIZE0, uint64_t DATABLOCKSIZE1, uint64_t PADDINGSIZE0, uint64_t PADDINGSIZE1, uint64_t dataBlockAmount)
{
    // determining threadId
    uint64_t idx = ((gridDim.x * gridDim.y * blockIdx.z) + (gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z)
        + ((blockDim.x * blockDim.y) * threadIdx.z + (blockDim.x * threadIdx.y) + threadIdx.x);

    // determining blocksize and padding size
    uint64_t dataBlockSize = DATABLOCKSIZE0;
    uint64_t paddingSize = PADDINGSIZE0;
    if (DATABLOCKSIZE1 > 0 && idx == dataBlockAmount - 1)
    {
        dataBlockSize = DATABLOCKSIZE1;
        paddingSize = PADDINGSIZE1;
    }

    // initial address in D_C per thread
    uint64_t x1 = DATABLOCKSIZE0 * idx;

    // initial address in D_P per thread
    uint64_t x2 = (DATABLOCKSIZE0 + PADDINGSIZE0) * idx;

    if (idx < dataBlockAmount)
    {
        // cpy chars from orginal chars address to padded address
        memcpy(&D_P[x2], &D_C[x1], dataBlockSize);

        //  first time padding, padding 1000 0000
        D_P[x2 + dataBlockSize] = 0x80;

        // second time padding, padding 0000 0000, (paddingsize -9) times
        for (int i = 1; i <= paddingSize - 9; i++)
        {
            D_P[x2 + dataBlockSize + i] = 0x00;
        }

        // third time padding, padding data block length
        for (int i = 1; i <= 8; i++)
        {
            D_P[x2 + dataBlockSize + paddingSize - i] = (unsigned char)((8 * dataBlockSize) >> (i - 1) * 8);
        }
    }
}

// transform 4 unsigned char to 32-bit unsiged int
__global__ void unsignedCharToUnsignedInt(const unsigned char *D_P, uint32_t *D_T, uint64_t dataBlockAmount, uint64_t groupsPerThread_1, uint64_t groupsPerThread_2)
{
    // determining threadId
    uint64_t idx = ((gridDim.x * gridDim.y * blockIdx.z) 
                    + (gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z)
                    + ((blockDim.x * blockDim.y) * threadIdx.z + (blockDim.x * threadIdx.y) + threadIdx.x);

    // initial address in D_P per thread
    uint64_t x1 = idx * 64 * groupsPerThread_1;

    // initial address in D_T per thread
    uint64_t x2 = idx * 16 * groupsPerThread_1;

    // the cycle counts of transform from unsigned char to unsigned int per thread
    uint64_t cycle_counts = groupsPerThread_1;
    if (idx == dataBlockAmount - 1) cycle_counts = groupsPerThread_2;

    // transform
    if (idx < dataBlockAmount)
    { 
        for (uint64_t i = 0; i < cycle_counts; i++)
        {
            D_T[x2 + 0 + 16 * i] = (D_P[x1 + 0 + 64 * i] << 24) + (D_P[x1 + 1 + 64 * i] << 16) + (D_P[x1 + 2 + 64 * i] << 8) + D_P[x1 + 3 + 64 * i];
            D_T[x2 + 1 + 16 * i] = (D_P[x1 + 4 + 64 * i] << 24) + (D_P[x1 + 5 + 64 * i] << 16) + (D_P[x1 + 6 + 64 * i] << 8) + D_P[x1 + 7 + 64 * i];
            D_T[x2 + 2 + 16 * i] = (D_P[x1 + 8 + 64 * i] << 24) + (D_P[x1 + 9 + 64 * i] << 16) + (D_P[x1 + 10 + 64 * i] << 8) + D_P[x1 + 11 + 64 * i];
            D_T[x2 + 3 + 16 * i] = (D_P[x1 + 12 + 64 * i] << 24) + (D_P[x1 + 13 + 64 * i] << 16) + (D_P[x1 + 14 + 64 * i] << 8) + D_P[x1 + 15 + 64 * i];
            D_T[x2 + 4 + 16 * i] = (D_P[x1 + 16 + 64 * i] << 24) + (D_P[x1 + 17 + 64 * i] << 16) + (D_P[x1 + 18 + 64 * i] << 8) + D_P[x1 + 19 + 64 * i];
            D_T[x2 + 5 + 16 * i] = (D_P[x1 + 20 + 64 * i] << 24) + (D_P[x1 + 21 + 64 * i] << 16) + (D_P[x1 + 22 + 64 * i] << 8) + D_P[x1 + 23 + 64 * i];
            D_T[x2 + 6 + 16 * i] = (D_P[x1 + 24 + 64 * i] << 24) + (D_P[x1 + 25 + 64 * i] << 16) + (D_P[x1 + 26 + 64 * i] << 8) + D_P[x1 + 27 + 64 * i];
            D_T[x2 + 7 + 16 * i] = (D_P[x1 + 28 + 64 * i] << 24) + (D_P[x1 + 29 + 64 * i] << 16) + (D_P[x1 + 30 + 64 * i] << 8) + D_P[x1 + 31 + 64 * i];
            D_T[x2 + 8 + 16 * i] = (D_P[x1 + 32 + 64 * i] << 24) + (D_P[x1 + 33 + 64 * i] << 16) + (D_P[x1 + 34 + 64 * i] << 8) + D_P[x1 + 35 + 64 * i];
            D_T[x2 + 9 + 16 * i] = (D_P[x1 + 36 + 64 * i] << 24) + (D_P[x1 + 37 + 64 * i] << 16) + (D_P[x1 + 38 + 64 * i] << 8) + D_P[x1 + 39 + 64 * i];
            D_T[x2 + 10 + 16 * i] = (D_P[x1 + 40 + 64 * i] << 24) + (D_P[x1 + 41 + 64 * i] << 16) + (D_P[x1 + 42 + 64 * i] << 8) + D_P[x1 + 43 + 64 * i];
            D_T[x2 + 11 + 16 * i] = (D_P[x1 + 44 + 64 * i] << 24) + (D_P[x1 + 45 + 64 * i] << 16) + (D_P[x1 + 46 + 64 * i] << 8) + D_P[x1 + 47 + 64 * i];
            D_T[x2 + 12 + 16 * i] = (D_P[x1 + 48 + 64 * i] << 24) + (D_P[x1 + 49 + 64 * i] << 16) + (D_P[x1 + 50 + 64 * i] << 8) + D_P[x1 + 51 + 64 * i];
            D_T[x2 + 13 + 16 * i] = (D_P[x1 + 52 + 64 * i] << 24) + (D_P[x1 + 53 + 64 * i] << 16) + (D_P[x1 + 54 + 64 * i] << 8) + D_P[x1 + 55 + 64 * i];
            D_T[x2 + 14 + 16 * i] = (D_P[x1 + 56 + 64 * i] << 24) + (D_P[x1 + 57 + 64 * i] << 16) + (D_P[x1 + 58 + 64 * i] << 8) + D_P[x1 + 59 + 64 * i];
            D_T[x2 + 15 + 16 * i] = (D_P[x1 + 60 + 64 * i] << 24) + (D_P[x1 + 61 + 64 * i] << 16) + (D_P[x1 + 62 + 64 * i] << 8) + D_P[x1 + 63 + 64 * i];
        }
    }
}

// extending 16 32-bit integers to 64 32-bit integers
__global__ void extending(uint32_t *D_T, uint32_t *D_E, uint64_t dataBlockAmount, uint64_t groupsPerThread_1, uint64_t groupsPerThread_2)
{
    // determining threadId
    uint64_t idx = ((gridDim.x * gridDim.y * blockIdx.z) + (gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z)
                    + ((blockDim.x * blockDim.y) * threadIdx.z + (blockDim.x * threadIdx.y) + threadIdx.x);

    // initial address in D_T per thread
    uint64_t x1 = idx * 16 * groupsPerThread_1;

    // initial address in D_E per thread
    uint64_t x2 = idx * 64 * groupsPerThread_1;

    uint64_t cycle_counts = groupsPerThread_1;
    if (idx == dataBlockAmount - 1) cycle_counts = groupsPerThread_2;

    if (idx < dataBlockAmount)
    {
        for (uint64_t i = 0; i < cycle_counts; i++)
        {
            D_E[x2 + 64 * i + 0] = D_T[x1 + 16 * i + 0];
            D_E[x2 + 64 * i + 1] = D_T[x1 + 16 * i + 1];
            D_E[x2 + 64 * i + 2] = D_T[x1 + 16 * i + 2];
            D_E[x2 + 64 * i + 3] = D_T[x1 + 16 * i + 3];
            D_E[x2 + 64 * i + 4] = D_T[x1 + 16 * i + 4];
            D_E[x2 + 64 * i + 5] = D_T[x1 + 16 * i + 5];
            D_E[x2 + 64 * i + 6] = D_T[x1 + 16 * i + 6];
            D_E[x2 + 64 * i + 7] = D_T[x1 + 16 * i + 7];
            D_E[x2 + 64 * i + 8] = D_T[x1 + 16 * i + 8];
            D_E[x2 + 64 * i + 9] = D_T[x1 + 16 * i + 9];
            D_E[x2 + 64 * i + 10] = D_T[x1 + 16 * i + 10];
            D_E[x2 + 64 * i + 11] = D_T[x1 + 16 * i + 11];
            D_E[x2 + 64 * i + 12] = D_T[x1 + 16 * i + 12];
            D_E[x2 + 64 * i + 13] = D_T[x1 + 16 * i + 13];
            D_E[x2 + 64 * i + 14] = D_T[x1 + 16 * i + 14];
            D_E[x2 + 64 * i + 15] = D_T[x1 + 16 * i + 15];
            for (uint64_t j = 16; j < 64; j++)
            {
                D_E[x2 + j + 64 * i] = SSigma_1(D_E[x2 + j + 64 * i - 2]) + D_E[x2 + j + 64 * i - 7] + SSigma_0(D_E[x2 + j + 64 * i - 15]) + D_E[x2 + j + 64 * i - 16];
                D_E[x2 + j + 64 * i] = D_E[x2 + j + 64 * i] & 0xFFFFFFFF;
            }
        }
    }
}

// updating hash value
__global__ void updatingHashValue(const uint32_t *D_E, uint32_t *D_H, uint64_t DATABLOCKSIZE0, uint64_t DATABLOCKSIZE1, uint64_t PADDINGSIZE0, uint64_t PADDINGSIZE1, uint64_t dataBlockAmount, uint64_t layer, bool oddDataBlockAmount, uint64_t hashValuePosition)
{
    uint64_t idx = ((gridDim.x * gridDim.y * blockIdx.z) + (gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z)
        + ((blockDim.x * blockDim.y) * threadIdx.z + (blockDim.x * threadIdx.y) + threadIdx.x);

    // determining blocksize and padding size
    uint64_t dataBlockSize = DATABLOCKSIZE0;
    uint64_t paddingSize = PADDINGSIZE0;
    if (DATABLOCKSIZE1 > 0 && idx == dataBlockAmount - 1)
    {
        dataBlockSize = DATABLOCKSIZE1;
        paddingSize = PADDINGSIZE1;
    }

    // initial address in D_E per thread
    uint64_t x1 = (DATABLOCKSIZE0 + PADDINGSIZE0) * idx;

    // initial address in D_H per thread
    uint64_t x2 = 8 * idx;

    // determining the number of 64_byte_groups for per data block
    uint64_t groupsPerDataBlock = (dataBlockSize + paddingSize) / 64;

    // preprocess
    uint32_t t1, t2, h1, h2, h3, h4, h5, h6, h7, h8;

    if (idx < dataBlockAmount)
    {
        D_H[x2 + 0 + hashValuePosition] = h1 = 0x6a09e667;
        D_H[x2 + 1 + hashValuePosition] = h2 = 0xbb67ae85;
        D_H[x2 + 2 + hashValuePosition] = h3 = 0x3c6ef372;
        D_H[x2 + 3 + hashValuePosition] = h4 = 0xa54ff53a;
        D_H[x2 + 4 + hashValuePosition] = h5 = 0x510e527f;
        D_H[x2 + 5 + hashValuePosition] = h6 = 0x9b05688c;
        D_H[x2 + 6 + hashValuePosition] = h7 = 0x1f83d9ab;
        D_H[x2 + 7 + hashValuePosition] = h8 = 0x5be0cd19;
    }

    const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, \
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, \
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, \
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, \
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, \
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, \
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, \
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, \
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, \
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, \
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, \
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, \
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, \
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, \
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, \
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2, \
    };

    // cycliclly updating hash value
    if (idx < dataBlockAmount)
    {
        for (uint32_t i = 0; i < groupsPerDataBlock; i++)
        {
            for (uint32_t j = 0; j < 64; j++)
            {
                t1 = (h8 + LSigma_1(h5) + Conditional(h5, h6, h7) + K[j] + D_E[x1 + j + 64 * i]) & 0xFFFFFFFF;
                t2 = (LSigma_0(h1) + Majority(h1, h2, h3)) & 0xFFFFFFFF;
                h8 = h7;
                h7 = h6;
                h6 = h5;
                h5 = (h4 + t1) & 0xFFFFFFFF;
                h4 = h3;
                h3 = h2;
                h2 = h1;
                h1 = (t1 + t2) & 0xFFFFFFFF;
            }
            D_H[x2 + 0 + hashValuePosition] = (D_H[x2 + 0 + hashValuePosition] + h1) & 0xFFFFFFFF;
            D_H[x2 + 1 + hashValuePosition] = (D_H[x2 + 1 + hashValuePosition] + h2) & 0xFFFFFFFF;
            D_H[x2 + 2 + hashValuePosition] = (D_H[x2 + 2 + hashValuePosition] + h3) & 0xFFFFFFFF;
            D_H[x2 + 3 + hashValuePosition] = (D_H[x2 + 3 + hashValuePosition] + h4) & 0xFFFFFFFF;
            D_H[x2 + 4 + hashValuePosition] = (D_H[x2 + 4 + hashValuePosition] + h5) & 0xFFFFFFFF;
            D_H[x2 + 5 + hashValuePosition] = (D_H[x2 + 5 + hashValuePosition] + h6) & 0xFFFFFFFF;
            D_H[x2 + 6 + hashValuePosition] = (D_H[x2 + 6 + hashValuePosition] + h7) & 0xFFFFFFFF;
            D_H[x2 + 7 + hashValuePosition] = (D_H[x2 + 7 + hashValuePosition] + h8) & 0xFFFFFFFF;
            h1 = D_H[x2 + 0 + hashValuePosition];
            h2 = D_H[x2 + 1 + hashValuePosition];
            h3 = D_H[x2 + 2 + hashValuePosition];
            h4 = D_H[x2 + 3 + hashValuePosition];
            h5 = D_H[x2 + 4 + hashValuePosition];
            h6 = D_H[x2 + 5 + hashValuePosition];
            h7 = D_H[x2 + 6 + hashValuePosition];
            h8 = D_H[x2 + 7 + hashValuePosition];
        }
    }

    // when the number of hash vaule amount is odd, copy the last-1 hash value
    if (oddDataBlockAmount && (idx == dataBlockAmount - 1))
    {
        D_H[8 * dataBlockAmount + 0 + hashValuePosition] = D_H[8 * dataBlockAmount - 8 + hashValuePosition];
        D_H[8 * dataBlockAmount + 1 + hashValuePosition] = D_H[8 * dataBlockAmount - 7 + hashValuePosition];
        D_H[8 * dataBlockAmount + 2 + hashValuePosition] = D_H[8 * dataBlockAmount - 6 + hashValuePosition];
        D_H[8 * dataBlockAmount + 3 + hashValuePosition] = D_H[8 * dataBlockAmount - 5 + hashValuePosition];
        D_H[8 * dataBlockAmount + 4 + hashValuePosition] = D_H[8 * dataBlockAmount - 4 + hashValuePosition];
        D_H[8 * dataBlockAmount + 5 + hashValuePosition] = D_H[8 * dataBlockAmount - 3 + hashValuePosition];
        D_H[8 * dataBlockAmount + 6 + hashValuePosition] = D_H[8 * dataBlockAmount - 2 + hashValuePosition];
        D_H[8 * dataBlockAmount + 7 + hashValuePosition] = D_H[8 * dataBlockAmount - 1 + hashValuePosition];
    }
}

// little end to big end
__global__ void lend_to_bend(uint32_t *V, uint64_t threads)
{
    uint64_t idx = ((gridDim.x * gridDim.y * blockIdx.z) + (gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z)
        + ((blockDim.x * blockDim.y) * threadIdx.z + (blockDim.x * threadIdx.y) + threadIdx.x);

    // little end to big end
    if (idx < threads)
    {
        V[idx] = (V[idx] & 0x000000FFU) << 24 | (V[idx] & 0x0000FF00U) << 8 | (V[idx] & 0x00FF0000U) >> 8 | (V[idx] & 0xFF000000U) >> 24;
    }
}