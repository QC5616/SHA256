sha256_mht : sha256.c sha256OpenSSL.c sha256.cu
	gcc sha256.c -w -o a_sha256_c
	gcc sha256OpenSSL.c -w -lcrypto -o b_sha256.OpenSSL
	nvcc -arch=sm_75 sha256.cu -w -o c_sha256_cu
