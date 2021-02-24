#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdbool.h>

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// logic functions
#define ROTL(W, n) (((W << n) & 0xFFFFFFFF) | (W) >> (32 - (n)))
#define SHR(W, n) ((W >> n) & 0xFFFFFFFF)
#define Conditional(x, y, z) ((x & y) ^ ((~x) & z))
#define Majority(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define LSigma_0(x) (ROTL(x, 30) ^ ROTL(x, 19) ^ ROTL(x, 10))
#define LSigma_1(x) (ROTL(x, 26) ^ ROTL(x, 21) ^ ROTL(x, 7))
#define SSigma_0(x) (ROTL(x, 25) ^ ROTL(x, 14) ^ SHR(x, 3))
#define SSigma_1(x) (ROTL(x, 15) ^ ROTL(x, 13) ^ SHR(x, 10))

// the numbers of characters per reading file
uint64_t READSIZE = 10LL * 1024 * 1024 * 1024;

// the path of file
const char *FILEPATH = "D:\\CUDA\\SHA256\\q.txt";

// the size of a data block
uint64_t DATABLOCKSIZE[2] = {1024llu, 0LLU};

// recording time in seconds
double getTime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// 1. padding characters
void paddingChar(unsigned char *P, uint64_t dataBlockSize, uint64_t paddingSize);

// 2. transform 4 unsigned char to 1 32-bit unsigned int
void unsignedCharToUnsignedInt(const unsigned char *P, uint32_t *T, uint64_t N);

// 3. extending 16 32-bit integers to 64 32-bit integers
void extending(const uint32_t *d_T, uint32_t *d_E, uint32_t N);

// 4. updating hash value
void updatingHashValue(const uint32_t *d_E, uint32_t *d_H, uint64_t N, uint64_t dataBlockAmount, uint64_t a, bool oddDataBlockAmount);

// 5. little end to big end
uint32_t lend_to_bend(uint32_t value, uint64_t l);

// main function
int main(int argc, char *argv[])
{
    printf("\nComputing hash value on CPU.\n");

    // determining data block size
    uint64_t coef = 0;
    printf("Please enter DataBlock size coefficient in KB: ");
    scanf("%llu", &coef);
    DATABLOCKSIZE[0] = coef * 1024;
    if (DATABLOCKSIZE[0] > 600 * 1024 * 1024LLU) 
    {
        printf("The data block is too big!\n");
        exit(EXIT_FAILURE);
    }

    // set the start time
    double start, phase_1, end;
    start = getTime();

    // get the file size
    printf("have read file: %s\n", argv[1]);
    FILE *fin;
    fin = fopen(argv[1], "rb");
    if (!fin)
    {
        printf("reading file failed.\n");
        if (argc == 1)
            printf("please enter file name.\n");
        exit(EXIT_FAILURE);
    }
    fseek(fin, 0, SEEK_END);
    uint64_t fileSize = ftell(fin);
    rewind(fin);
    printf("the size of file: %llu Bytes\n", fileSize);

    // get the reading times
    if (DATABLOCKSIZE[0] > fileSize)  DATABLOCKSIZE[0] = fileSize;
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

    // computing hash value for 0 layer

    // 1. determining the the size of data block
    if (READSIZE % DATABLOCKSIZE[0] > 0)
    {
        DATABLOCKSIZE[1] = READSIZE % DATABLOCKSIZE[0];
    }

    // 2. get the number of characters for padding
    uint64_t PADDINGSIZE[2] = {0LLU, 0LLU};
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

    // 3. get the number of data block
    uint64_t dataBlockAmount = fileSize / DATABLOCKSIZE[0];
    if (fileSize % DATABLOCKSIZE[0] > 0)
        dataBlockAmount++;

    // 4. determining the parity of data block amount
    bool oddDataBlockAmount = false;
    if (dataBlockAmount % 2 != 0)
        oddDataBlockAmount = true;

    // 5. get the number of hash value
    uint64_t hashValueAmount = dataBlockAmount;
    if (oddDataBlockAmount && layers > 1)
        hashValueAmount++;
    uint64_t hashValueAmountArray[layers];
    hashValueAmountArray[0] = hashValueAmount;

    // pre-assign the size of block and the number of characters for padding
    uint64_t dataBlockSize = DATABLOCKSIZE[0];
    uint64_t paddingSize = PADDINGSIZE[0];

    // storing the data after padding
    char *P = (char *)malloc(dataBlockSize + paddingSize);

    //  storing the data after transform
    uint32_t *T = (uint32_t *)malloc(dataBlockSize + paddingSize);

    // storing the data after extending
    uint32_t *E = (uint32_t *)malloc(4 * (dataBlockSize + paddingSize));

    // storing the hash value of the data
    uint32_t *V[layers];
    V[0] = (uint32_t *)malloc(hashValueAmount * 8 * sizeof(uint32_t));

    // cyclically updating data block's hash value
    for (uint64_t a = 0; a < dataBlockAmount; a++)
    {
        // handle the particular data block
        if (a == dataBlockAmount - 1 && DATABLOCKSIZE[1] > 0)
        {
            dataBlockSize = DATABLOCKSIZE[1];
            paddingSize = PADDINGSIZE[1];
            P = (char *)realloc(P, dataBlockSize + paddingSize);
            T = (uint32_t *)realloc(T, dataBlockSize + paddingSize);
            E = (uint32_t *)realloc(E, 4 * (dataBlockSize + paddingSize));
        }

        // 1. get characters from input data stream
        fread(P, 1, dataBlockSize, fin);

        // 2. determining the number of characters for padding
        paddingChar((unsigned char *)P, dataBlockSize, paddingSize);

        // 3. transform 4 unsigned char to 1 32-bit unsigned int and performing N times
        unsignedCharToUnsignedInt((unsigned char *)P, T, (dataBlockSize + paddingSize) / 64);

        // 4. extending 16 32-bit integers to 64 32-bit integers and performing N times
        extending(T, E, (dataBlockSize + paddingSize) / 64);

        // 5.updating hash value and performing N times
        updatingHashValue(E, V[0], (dataBlockSize + paddingSize) / 64, dataBlockAmount, a, oddDataBlockAmount);
    }

    // transform little end to big end
    for (uint64_t i = 0; i < hashValueAmountArray[0] * 8; ++i)
    {
        V[0][i] = lend_to_bend(V[0][i], 0);
    }

    // recording phase 1 time
    phase_1 = getTime();

    // computing hash value for 1 to (layers-1) layer

    // pre-assign the size of data block and the number of characters for padding
    dataBlockSize = 64LLU;
    paddingSize = 64LLU;

    // storing the data after padding
    P = (char *)realloc(P, dataBlockSize + paddingSize);

    //  storing the data after transform
    T = (uint32_t *)realloc(T, dataBlockSize + paddingSize);

    // storing the data after extending
    E = (uint32_t *)realloc(E, 4 * (dataBlockSize + paddingSize));

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
        V[l] = (uint32_t *)malloc(hashValueAmount * 8 * sizeof(uint32_t));

        // computing hash value for per layer
        for (uint64_t a = 0; a < dataBlockAmount; a++)
        {
            // 1. fetch data from the previous hash value
            memcpy(P, &V[l - 1][a * 16], dataBlockSize);

            // 2. padding characters
            paddingChar((unsigned char *)P, dataBlockSize, paddingSize);

            // 3. transform 4 unsigned char to 32-bit unsigned int
            unsignedCharToUnsignedInt((unsigned char *)P, T, (dataBlockSize + paddingSize) / 64);

            // 4. extending 16 32-bit integers to 64 32-bit integersX
            extending(T, E, (dataBlockSize + paddingSize) / 64);

            // 5.updating hash value
            updatingHashValue(E, V[l], (dataBlockSize + paddingSize) / 64, dataBlockAmount, a, oddDataBlockAmount);
        }

        // transform little end to big end
        for (uint64_t i = 0; i < hashValueAmountArray[l] * 8; ++i)
        {
            V[l][i] = lend_to_bend(V[l][i], l);
        }

        // printf("layer  = %lu\n", l);
        // for (uint64_t i = 0; i < hashValueAmountArray[l] * 8; ++i)
        // {
        //     if (1) 
        //     {
        //         printf("V[%lu][%lu] = %08x\n", l, i, V[l][i]);
        //     }
            
        // }
        

    }

    // set the end time
    end = getTime();

    // present the merkle root
    printf("the merkle root: ");
    unsigned char *Temp = (unsigned char *)&V[layers - 1][0];
    for (uint64_t j = 0; j < 32; ++j)
    {
        printf("%02x", Temp[j]);
    }
    printf("\n");

    // for (uint64_t i = 0; i < layers; ++i)
    // {
    //     printf("the %lu layer hash value:\n", i);
    //     for (uint64_t k = 0; k < hashValueAmountArray[i]; ++k)
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
    free(T);
    free(E);
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

// 1. padding characters
void paddingChar(unsigned char *P, uint64_t dataBlockSize, uint64_t paddingSize)
{
    //  first padding 1000 0000
    P[dataBlockSize] = 0x80;

    // second padding 0000 0000
    for (int i = 1; i <= paddingSize - 9; i++)
    {
        P[dataBlockSize + i] = 0x00;
    }

    // third padding data length presented in binary
    for (int i = 1; i <= 8; i++)
    {
        P[dataBlockSize + paddingSize - i] = (unsigned char)((8 * dataBlockSize) >> (i - 1) * 8);
    }
}

// 2. transform 4 unsigned char to 32-bit unsigned int
void unsignedCharToUnsignedInt(const unsigned char *P, uint32_t *T, uint64_t N)
{
    for (uint64_t i = 0; i < N; i++)
    {
        T[0 + 16 * i] = (P[0 + 64 * i] << 24) + (P[1 + 64 * i] << 16) + (P[2 + 64 * i] << 8) + P[3 + 64 * i];
        T[1 + 16 * i] = (P[4 + 64 * i] << 24) + (P[5 + 64 * i] << 16) + (P[6 + 64 * i] << 8) + P[7 + 64 * i];
        T[2 + 16 * i] = (P[8 + 64 * i] << 24) + (P[9 + 64 * i] << 16) + (P[10 + 64 * i] << 8) + P[11 + 64 * i];
        T[3 + 16 * i] = (P[12 + 64 * i] << 24) + (P[13 + 64 * i] << 16) + (P[14 + 64 * i] << 8) + P[15 + 64 * i];
        T[4 + 16 * i] = (P[16 + 64 * i] << 24) + (P[17 + 64 * i] << 16) + (P[18 + 64 * i] << 8) + P[19 + 64 * i];
        T[5 + 16 * i] = (P[20 + 64 * i] << 24) + (P[21 + 64 * i] << 16) + (P[22 + 64 * i] << 8) + P[23 + 64 * i];
        T[6 + 16 * i] = (P[24 + 64 * i] << 24) + (P[25 + 64 * i] << 16) + (P[26 + 64 * i] << 8) + P[27 + 64 * i];
        T[7 + 16 * i] = (P[28 + 64 * i] << 24) + (P[29 + 64 * i] << 16) + (P[30 + 64 * i] << 8) + P[31 + 64 * i];
        T[8 + 16 * i] = (P[32 + 64 * i] << 24) + (P[33 + 64 * i] << 16) + (P[34 + 64 * i] << 8) + P[35 + 64 * i];
        T[9 + 16 * i] = (P[36 + 64 * i] << 24) + (P[37 + 64 * i] << 16) + (P[38 + 64 * i] << 8) + P[39 + 64 * i];
        T[10 + 16 * i] = (P[40 + 64 * i] << 24) + (P[41 + 64 * i] << 16) + (P[42 + 64 * i] << 8) + P[43 + 64 * i];
        T[11 + 16 * i] = (P[44 + 64 * i] << 24) + (P[45 + 64 * i] << 16) + (P[46 + 64 * i] << 8) + P[47 + 64 * i];
        T[12 + 16 * i] = (P[48 + 64 * i] << 24) + (P[49 + 64 * i] << 16) + (P[50 + 64 * i] << 8) + P[51 + 64 * i];
        T[13 + 16 * i] = (P[52 + 64 * i] << 24) + (P[53 + 64 * i] << 16) + (P[54 + 64 * i] << 8) + P[55 + 64 * i];
        T[14 + 16 * i] = (P[56 + 64 * i] << 24) + (P[57 + 64 * i] << 16) + (P[58 + 64 * i] << 8) + P[59 + 64 * i];
        T[15 + 16 * i] = (P[60 + 64 * i] << 24) + (P[61 + 64 * i] << 16) + (P[62 + 64 * i] << 8) + P[63 + 64 * i];
    }
}

// 3. extending 16 32-bit integers to 64 32-bit integers
void extending(const uint32_t *T, uint32_t *E, uint32_t N)
{
    for (uint32_t i = 0; i < N; i++)
    {
        E[64 * i + 0] = T[16 * i + 0];
        E[64 * i + 1] = T[16 * i + 1];
        E[64 * i + 2] = T[16 * i + 2];
        E[64 * i + 3] = T[16 * i + 3];
        E[64 * i + 4] = T[16 * i + 4];
        E[64 * i + 5] = T[16 * i + 5];
        E[64 * i + 6] = T[16 * i + 6];
        E[64 * i + 7] = T[16 * i + 7];
        E[64 * i + 8] = T[16 * i + 8];
        E[64 * i + 9] = T[16 * i + 9];
        E[64 * i + 10] = T[16 * i + 10];
        E[64 * i + 11] = T[16 * i + 11];
        E[64 * i + 12] = T[16 * i + 12];
        E[64 * i + 13] = T[16 * i + 13];
        E[64 * i + 14] = T[16 * i + 14];
        E[64 * i + 15] = T[16 * i + 15];
        for (uint32_t j = 16; j < 64; j++)
        {
            E[j + 64 * i] = SSigma_1(E[j + 64 * i - 2]) + E[j + 64 * i - 7] + SSigma_0(E[j + 64 * i - 15]) + E[j + 64 * i - 16];
        }
    }
}

// 4. updating hash value
void updatingHashValue(const uint32_t *E, uint32_t *H, uint64_t N, uint64_t dataBlockAmount, uint64_t a, bool oddDataBlockAmount)
{
    // preprocess
    uint32_t t1, t2, h1, h2, h3, h4, h5, h6, h7, h8;

    H[8 * a + 0] = h1 = 0x6a09e667;
    H[8 * a + 1] = h2 = 0xbb67ae85;
    H[8 * a + 2] = h3 = 0x3c6ef372;
    H[8 * a + 3] = h4 = 0xa54ff53a;
    H[8 * a + 4] = h5 = 0x510e527f;
    H[8 * a + 5] = h6 = 0x9b05688c;
    H[8 * a + 6] = h7 = 0x1f83d9ab;
    H[8 * a + 7] = h8 = 0x5be0cd19;

    const uint32_t K[64] = {
        0x428a2f98,
        0x71374491,
        0xb5c0fbcf,
        0xe9b5dba5,
        0x3956c25b,
        0x59f111f1,
        0x923f82a4,
        0xab1c5ed5,
        0xd807aa98,
        0x12835b01,
        0x243185be,
        0x550c7dc3,
        0x72be5d74,
        0x80deb1fe,
        0x9bdc06a7,
        0xc19bf174,
        0xe49b69c1,
        0xefbe4786,
        0x0fc19dc6,
        0x240ca1cc,
        0x2de92c6f,
        0x4a7484aa,
        0x5cb0a9dc,
        0x76f988da,
        0x983e5152,
        0xa831c66d,
        0xb00327c8,
        0xbf597fc7,
        0xc6e00bf3,
        0xd5a79147,
        0x06ca6351,
        0x14292967,
        0x27b70a85,
        0x2e1b2138,
        0x4d2c6dfc,
        0x53380d13,
        0x650a7354,
        0x766a0abb,
        0x81c2c92e,
        0x92722c85,
        0xa2bfe8a1,
        0xa81a664b,
        0xc24b8b70,
        0xc76c51a3,
        0xd192e819,
        0xd6990624,
        0xf40e3585,
        0x106aa070,
        0x19a4c116,
        0x1e376c08,
        0x2748774c,
        0x34b0bcb5,
        0x391c0cb3,
        0x4ed8aa4a,
        0x5b9cca4f,
        0x682e6ff3,
        0x748f82ee,
        0x78a5636f,
        0x84c87814,
        0x8cc70208,
        0x90befffa,
        0xa4506ceb,
        0xbef9a3f7,
        0xc67178f2,
    };

    for (uint32_t i = 0; i < N; i++)
    {
        for (int j = 0; j < 64; j++)
        {
            t1 = h8 + LSigma_1(h5) + Conditional(h5, h6, h7) + K[j] + E[j + 64 * i];
            t2 = LSigma_0(h1) + Majority(h1, h2, h3);
            h8 = h7;
            h7 = h6;
            h6 = h5;
            h5 = h4 + t1;
            h4 = h3;
            h3 = h2;
            h2 = h1;
            h1 = t1 + t2;
        }
        H[8 * a + 0] = (H[8 * a + 0] + h1) & 0xFFFFFFFF;
        H[8 * a + 1] = (H[8 * a + 1] + h2) & 0xFFFFFFFF;
        H[8 * a + 2] = (H[8 * a + 2] + h3) & 0xFFFFFFFF;
        H[8 * a + 3] = (H[8 * a + 3] + h4) & 0xFFFFFFFF;
        H[8 * a + 4] = (H[8 * a + 4] + h5) & 0xFFFFFFFF;
        H[8 * a + 5] = (H[8 * a + 5] + h6) & 0xFFFFFFFF;
        H[8 * a + 6] = (H[8 * a + 6] + h7) & 0xFFFFFFFF;
        H[8 * a + 7] = (H[8 * a + 7] + h8) & 0xFFFFFFFF;
        h1 = H[8 * a + 0];
        h2 = H[8 * a + 1];
        h3 = H[8 * a + 2];
        h4 = H[8 * a + 3];
        h5 = H[8 * a + 4];
        h6 = H[8 * a + 5];
        h7 = H[8 * a + 6];
        h8 = H[8 * a + 7];
    }

    // when the number of hash vaule amount is odd, copy the last-1 hash value
    if (oddDataBlockAmount && (dataBlockAmount != 1) && (a == dataBlockAmount - 1))
    {
        H[8 * dataBlockAmount + 0] = H[8 * dataBlockAmount - 8];
        H[8 * dataBlockAmount + 1] = H[8 * dataBlockAmount - 7];
        H[8 * dataBlockAmount + 2] = H[8 * dataBlockAmount - 6];
        H[8 * dataBlockAmount + 3] = H[8 * dataBlockAmount - 5];
        H[8 * dataBlockAmount + 4] = H[8 * dataBlockAmount - 4];
        H[8 * dataBlockAmount + 5] = H[8 * dataBlockAmount - 3];
        H[8 * dataBlockAmount + 6] = H[8 * dataBlockAmount - 2];
        H[8 * dataBlockAmount + 7] = H[8 * dataBlockAmount - 1];
    }
}

// 5. little end to big end
uint32_t lend_to_bend(uint32_t value, uint64_t l)
{   
    return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 | (value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24;
}