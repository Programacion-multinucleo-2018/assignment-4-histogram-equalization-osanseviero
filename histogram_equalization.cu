// Implements image box blur using GPU.
// Compile with nvcc histogram_equalization.cu -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -std=c++11
// Used https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf to guide the solution

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


__global__ void create_histogram(unsigned char* input, int* histogram, int width, int height, int step) {
     //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int tid = yIndex * step + xIndex;
    const int histogram_index = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ int block_histogram[256];

    // Initialize histogram
    if(histogram_index < 256) {
        block_histogram[histogram_index] = 0;
    }
    __syncthreads();

    // Add 1 to the count for every element
    if((xIndex < width) && (yIndex < height)) {
        atomicAdd(&block_histogram[input[tid]], 1);  
    }
    __syncthreads();

    // Copy back to global memory
    if(histogram_index < 256) {
        atomicAdd(&histogram[histogram_index], block_histogram[histogram_index]); 
    }
}

__global__ void normalize_histogram(int* histogram, int* normalized_histogram, int width, int height, int step) {
    int histogram_idx = threadIdx.x + threadIdx.y * blockDim.x;
    if(histogram_idx < 256) {
        long int accumulated = 0;
        for(int i = 0; i <= histogram_idx; i++) {
            accumulated += histogram[i];
        }
        normalized_histogram[histogram_idx] = accumulated*255/(width*height);
    }
}

__global__ void equalizer(unsigned char* input, int* normalized_histogram, unsigned char* output, int width, int height, int step) {
     //2D Index of current thread
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    const int tid = yIndex * step + xIndex;

    if ((xIndex < width) && (yIndex < height)){
        output[tid] = normalized_histogram[input[tid]];
    }

}

void histogram_equalizer(const cv::Mat& input, cv::Mat& output) {
    unsigned char *d_input, *d_output;
    int* histogram, *normalized_histogram;

    // Allocation bytes
    const int bytes = input.step * input.rows;
    const int histogram_bytes = sizeof(int)*256;

    //Allocate device memory
    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMalloc((void**)&histogram, histogram_bytes);
    cudaMalloc((void**)&normalized_histogram, histogram_bytes);

    //Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input, input.ptr(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output.ptr(), bytes, cudaMemcpyHostToDevice);

    //Specify block size
    const dim3 block(32, 32);

    //Calculate grid size to cover the whole image
    const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));

    // Apply histogram equalization
    auto start_cpu =  std::chrono::high_resolution_clock::now();
    create_histogram<<<grid, block>>>(d_input, histogram, input.cols, input.rows, input.step);
    normalize_histogram<<<1, block>>>(histogram, normalized_histogram, input.cols, input.rows, input.step);
    equalizer<<<grid, block>>>(d_input, histogram, d_output, input.cols, input.rows, input.step);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("histogram equalization elapsed %f ms\n", duration_ms.count());

    // Sync
    cudaDeviceSynchronize();

     // Copy memory from device to host
    cudaMemcpy(output.ptr(), d_output,bytes,cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char *argv[]) {
    // Read input image
    std::string imagePath = "Images/dog3.jpeg";

    // Load input image
    cv::Mat inputImage = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
    cv::Mat output(inputImage.rows, inputImage.cols, CV_8UC1);

    // Convert to grayscale
    cv::Mat grayImage;
    cvtColor(inputImage, grayImage, CV_BGR2GRAY);

    // Use histogram equalizer
    histogram_equalizer(grayImage, output);

    // Display input and output images
    namedWindow("Input", cv::WINDOW_NORMAL);
    namedWindow("Output", cv::WINDOW_NORMAL);

    cv::resizeWindow("Input", 800, 600);
    cv::resizeWindow("Output", 800, 600);

    imshow("Input", grayImage);
    imshow("Output", output);

    cv::waitKey();

    return 0;
}