// Implements image box blur using CPU.
// Compile with g++ histogram_equalization.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -std=c++11
// Used https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf to guide the solution

#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void histogram_equalizer(const cv::Mat& input, cv::Mat& output) {
    // Total pixels
    long int total_pixels = input.step*input.rows;

    // Array to count number of pixels with certain intensity (0 to 255)
    int histogram[256] = {0};

    // Calculate histogram of image (# of pixels with intensity n)
    short int pixel_intensity;
    for(int i=0; i<total_pixels; i++) {
        pixel_intensity = input.data[i];
        histogram[pixel_intensity]++;
    }

    // Calculate normalized histogram
    long int accumulated = 0;
    for(int i=0; i<256; i++) {  
        // Get number of pixels with intensity i (cumulative)
        accumulated += histogram[i];

        // Multiply by the max intensity-1 (256-1) and divide by total pixels
        histogram[i] = accumulated*255/total_pixels;
    }

    // Create output image based on normalized histogram
    std::cout << total_pixels << std::endl;
    for(int i=0; i<total_pixels; i++) {
        output.data[i] = histogram[input.data[i]];
    }
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
    auto start_cpu =  std::chrono::high_resolution_clock::now();
    histogram_equalizer(grayImage, output);
    auto end_cpu =  std::chrono::high_resolution_clock::now();

    // Measure total time
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    printf("histogram equalization elapsed %f ms\n", duration_ms.count());
    
    // Display input and output images
    namedWindow("Input", cv::WINDOW_NORMAL);
    namedWindow("Output", cv::WINDOW_NORMAL);

    imshow("Input", grayImage);
    imshow("Output", output);

    cv::waitKey();

    return 0;
}