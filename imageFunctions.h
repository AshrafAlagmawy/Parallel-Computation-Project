/*
 This project demonstrates the use of parallel processing techniques and MPI
 for efficient image processing tasks, offering users a flexible and interactive interface
 to apply various filters and operations to images.
*/

#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * @brief Load an image from the specified file path.
 *
 * @param imagePath The file path of the image.
 * @param image Output parameter to store the loaded image.
 */
void loadImage(const string& imagePath, Mat& image);

/**
 * @brief Convert the input image to grayscale.
 *
 * @param image The input image to be converted.
 */
void convertToGrayscale(Mat& image);

/**
 * @brief Broadcast the dimensions of the input image to all processes.
 *
 * @param image The input image whose dimensions need to be broadcasted.
 * @param rows Output parameter to store the number of rows in the image.
 * @param cols Output parameter to store the number of columns in the image.
 * @param my_rank The rank of the calling process.
 */
void broadcastImageDimensions(Mat& image, int& rows, int& cols, int my_rank);

/**
 * for all the comming function
 *
 * @param imagePath The file path of the input image.
 * @param my_rank The rank of the calling process.
 * @param comm_sz The total number of processes.
 */

void histogramEqualization(const string& imagePath, int my_rank, int comm_sz);

void histogramEqualization3channels(const string& imagePath, int my_rank, int comm_sz);

void applyGaussianBlur(const string& imagePath, int my_rank, int comm_sz);

void edgeDetection(const string& imagePath, int my_rank, int comm_sz);

void imageRotation(const string& imagePath, int my_rank, int comm_sz);

void imageScaling(const string& imagePath, int my_rank, int comm_sz);

void colorSpaceConversion(const string& imagePath, int my_rank, int comm_sz);

void globalThresholding(const string& imagePath, int my_rank, int comm_sz);

void localThresholding(const string& imagePath, int my_rank, int comm_sz);

void compressImage(const string& imagePath, int my_rank, int comm_sz);

void median(const string& imagePath, int my_rank, int comm_sz);