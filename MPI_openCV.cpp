/* This project is an image processing application that applies various filtersand operations to images
 using parallel processing with MPI (Message Passing Interface) in C++. */

#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include "imageFunctions.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    string imagePath = "C:\\Users\\MG store\\Pictures\\girl.jpg";
    int choice;

    if (my_rank == 0) {
        cout << "__________________________________________________________________________" << endl;
        cout << "                     Image Processing Fillters in Parallel                " << endl;
        cout << "__________________________________________________________________________" << endl;

        cout << "\n\n 1 - Gaussian Blur \n 2 - Edge Detection \n 3 - Image Rotation \n 4 - Image Scaling \n 5 - Histogram Equalization \n 6 - Color Space Conversion \n 7 - Global Thresholding\n 8 - Local Thresholding \n 9 - Image Compression \n 10 - Median" << endl;
        cout << "\n Please, choose the filter number you need" << endl << endl;
        cin >> choice;
    }

    MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (choice < 1 || choice > 10) {
        if (my_rank == 0) {
            cout << "Invalid choice. Exiting..." << endl;
        }
        MPI_Finalize();
        return 0;
    }

    switch (choice)
    {
    case 1:
        applyGaussianBlur(imagePath, my_rank, comm_sz);
        break;
    case 2:
        edgeDetection(imagePath, my_rank, comm_sz);
        break;
    case 3:
        imageRotation(imagePath, my_rank, comm_sz);
        break;
    case 4:
        imageScaling(imagePath, my_rank, comm_sz);
        break;
    case 5:
          histogramEqualization(imagePath, my_rank, comm_sz);
        break;
    case 6:
        colorSpaceConversion(imagePath, my_rank, comm_sz);
        break;
    case 7:
        globalThresholding(imagePath, my_rank, comm_sz);
        break;
    case 8:
        localThresholding(imagePath, my_rank, comm_sz);
        break;
    case 9:
        compressImage(imagePath, my_rank, comm_sz);
        break;
    case 10:
        median(imagePath, my_rank, comm_sz);
        break;
    default:
        break;
    }

    MPI_Finalize();
    return 0;
}