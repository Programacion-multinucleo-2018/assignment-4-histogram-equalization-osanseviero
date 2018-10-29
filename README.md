# Assignment 4: Histogram Equalization

Assignment No 4 for the multi-core programming course. Implement histogram equalization for a gray scale image in CPU and GPU. The result of applying the algorithm to an image with low contrast can be seen in Figure 1:

![Figure 1](Images/histogram_equalization.png)
<br/>Figure 1: Expected Result.

The programs have to do the following:

1. Using Opencv, load and image and convert it to grayscale.
2. Calculate de histogram of the image.
3. Calculate the normalized sum of the histogram.
4. Create an output image based on the normalized histogram.
5. Display both the input and output images.

Test your code with the different images that are included in the *Images* folder. Include the average calculation time for both the CPU and GPU versions, as well as the speedup obtained, in the Readme.

Rubric:

1. Image is loaded correctly.
2. The histogram is calculated correctly using atomic operations.
3. The normalized histogram is correctly calculated.
4. The output image is correctly calculated.
5. For the GPU version, used shared memory where necessary.
6. Both images are displayed at the end.
7. Calculation times and speedup obtained are incuded in the Readme.

**Grade: 100**

## Data
The following table shows the time of execution for a 5760 × 3840 image. 

|CPU time (ms)| GPU time (ms) |
|--|--|
| 106.069519 |0.025589  |
| 105.909813 | 0.021552 |
| 106.003456|0.023438  |
| 105.987000 |0.024585  |
| 106.028008 |0.025731  |
| 106.146461 |0.025406  |
| 106.079681 | 0.024107 |
| 106.458138 |0.024169  |
| 106.004974 | 0.024668 |
| 106.103203 |0.024419  |
| 106.019058 |0.023254  |
| 106.132751 |0.023619  |
| 106.027618 |0.023725  |
| 106.679283 | 0.024272  |
| 106.110374 | 0.025185  |
| 106.011620 | 0.032576  |
| 105.973312 | 0.024799 |

Average CPU Time: 106.10260405882ms

Average GPU Time: 0.421094ms

Speedup: 251.968928692
