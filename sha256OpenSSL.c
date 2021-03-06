#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdint.h>
#include <openssl/sha.h>

// the numbers of characters per reading file
uint64_t READSIZE = 20LL * 1024 * 1024 * 1024;

// the path of file
const char *FILEPATH = "D:\\CUDA\\SHA256\\q.txt";

// the size of a data block
uint64_t DATABLOCKSIZE[2] = {0llu, 0LLU};

// 1. recording time in seconds
double getTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
};

// main function
int main(int argc, char *argv[])
{
    printf("\nComputing hash value on CPU_OpenSSL.\n");

    // determining data block size
    uint64_t coef = 0;
    printf("Please enter the coefficient of the size of the data block in KB: ");
    scanf("%llu", &coef);

    // set the start time
    double start, phase_1, end;
    start = getTime();

    // get the file size
    printf("Have read file: %s\n", argv[1]);
    FILE *fin;
    fin = fopen(argv[1], "rb");
    if (!fin)
    {
        printf("Reading file failed.\n");
        if (argc == 1)
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
        if (DATABLOCKSIZE[0] > fileSize)
        {
            printf("Data block is too big.");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        DATABLOCKSIZE[0] = fileSize;
    }
    if (fileSize % DATABLOCKSIZE[0] > 0)
    {
        DATABLOCKSIZE[1] = fileSize % DATABLOCKSIZE[0];
    }

    // get the reading times
    if (fileSize < READSIZE)
        READSIZE = fileSize;
    uint64_t readTimes = fileSize / READSIZE;
    if (fileSize % READSIZE > 0)
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

    // 1. get the number of data block
    uint64_t dataBlockAmount = fileSize / DATABLOCKSIZE[0];
    if (fileSize % DATABLOCKSIZE[0] > 0)
        dataBlockAmount++;

    // 2. determining the parity of data block amount
    bool oddDataBlockAmount = false;
    if (dataBlockAmount % 2 != 0)
        oddDataBlockAmount = true;

    // 3. get the number of hash value
    uint64_t hashValueAmount = dataBlockAmount;
    if (oddDataBlockAmount && layers > 1)
        hashValueAmount++;
    uint64_t hashValueAmountArray[layers];
    hashValueAmountArray[0] = hashValueAmount;

    // 4. pre-assign the size of block and the number of characters for padding
    uint64_t dataBlockSize = DATABLOCKSIZE[0];

    // storing the data after padding
    char *P = (char *)malloc(dataBlockSize);

    // stroung the hash vaule of the data using char array
    unsigned char *V[layers];
    V[0] = (unsigned char *)malloc(hashValueAmount * 32);

    char tmp[3] = {0};
    unsigned char md[32] = {0};

    // cyclically updating data block's hash value
    for (uint64_t i = 0; i < dataBlockAmount; ++i)
    {
        // handle the particular data block
        if (i == dataBlockAmount - 1 && DATABLOCKSIZE[1] > 0)
        {
            dataBlockSize = DATABLOCKSIZE[1];
            P = (char *)realloc(P, dataBlockSize);
        }

        // 1. get characters from input data stream
        fread(P, dataBlockSize, 1, fin);

        // 2. computing hash value by openssl
        SHA256((unsigned char *)P, dataBlockSize, md);

        // 3. stroing data block hash value
        memcpy(&(V[0][i * 32]), md, 32);
    }

    // when the number of hash vaule amount is odd, copy the hash value
    if (oddDataBlockAmount && (layers > 1))
    {
        for (uint64_t i = 0; i < 32; ++i)
        {
            V[0][32 * dataBlockAmount + i] = V[0][32 * dataBlockAmount - 32 + i];
        }
    }

    // recording phase 1 time
    phase_1 = getTime();

// *********************************************************************************
// ****************** Computing 2 ~ (layers - 1) layer hash value ******************
// *********************************************************************************

    // pre-assign the size of data block and the number of characters for padding
    dataBlockSize = 64LLU;

    // storing the data after padding
    P = (char *)realloc(P, dataBlockSize);

    // cyclically computing hash value
    for (uint64_t l = 1; l < layers; l++)
    {
        // update the number of data block in the current layer
        dataBlockAmount = hashValueAmountArray[l - 1] / 2;

        // updating the parity of data block amount for per layer
        oddDataBlockAmount = false;
        if (dataBlockAmount % 2 != 0)
            oddDataBlockAmount = true;

        // update the number of hash value for per layer
        hashValueAmount = dataBlockAmount;
        if (oddDataBlockAmount && l != layers - 1)
            hashValueAmount++;
        hashValueAmountArray[l] = hashValueAmount;

        // assign the storage space of hash value
        V[l] = (unsigned char *)malloc(hashValueAmount * 32);

        // computing hash value for per layer
        for (uint64_t i = 0; i < dataBlockAmount; ++i)
        {
            // 1. get data from the previous hash value
            memcpy(P, &V[l - 1][i * 64], dataBlockSize);

            // 2. computing hash value by openssl
            SHA256((unsigned char *)P, dataBlockSize, md);

            // 3. stroing data block hash value
            memcpy(&(V[l][i * 32]), md, 32);
        }

        // when the number of hash vaule amount is odd, copy the hash value
        if (oddDataBlockAmount && (l != layers - 1))
        {
            for (uint64_t i = 0; i < 32; ++i)
            {
                V[l][32 * dataBlockAmount + i] = V[l][32 * dataBlockAmount - 32 + i];
            }
        }
    }

    // set the end time
    end = getTime();

    // present the merkle root
    printf("The Merkle Root: ");
    unsigned char *Temp = (unsigned char *)&V[layers - 1][0];
    for (uint64_t i = 0; i < 32; ++i)
    {
        printf("%02x", Temp[i]);
    }
    printf("\n");

    // for (uint64_t i = 0; i < layers; ++i)
    // {
    //     printf("the %lu layer hash value:\n", i);
    //     for (uint64_t k = 0; k < hashValueAmountArray[i] && k < 2; ++k)
    //     {
    //         unsigned char *Temp = (unsigned char *)&V[i][8 * k];
    //         for (uint64_t j = 0; j < 32; ++j)
    //         {
    //             printf("%02x", Temp[j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // close file pointer and data pointer
    fclose(fin);
    free(P);
    for (uint64_t i = 0; i < layers; i++)
    {
        free(V[i]);
    }

    // show time consumption,
    // printf("phase 1 time consumption: %f s\n", phase_1 - start);
    // printf("phase 2 time consumption: %f s\n", end - phase_1);
    // printf("all phase time consumption: %f s\n\n", end - start);

    printf("%f\n", phase_1 - start);
    printf("%f\n", end - phase_1);
    printf("%f\n\n", end - start);

    return 0;
}