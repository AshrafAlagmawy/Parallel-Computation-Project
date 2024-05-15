#include "imageFunctions.h"
#include <iostream>
#include <mpi.h>

using namespace std;
using namespace cv;

void loadImage(const string& imagePath, Mat& image) {
    image = imread(imagePath);
    if (image.empty()) {
        cout << "Error: Image not loaded." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI execution if image loading fails
    }
}

void convertToGrayscale(Mat& image) {
    if (image.channels() > 1) {
        cvtColor(image, image, COLOR_BGR2GRAY);
    }
}

void broadcastImageDimensions(Mat& image, int& rows, int& cols, int my_rank) {
    // Broadcast image dimensions
    if (my_rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void distributeImage(const Mat& image, Mat& localImage, int my_rank, int comm_sz) {
    cout << "distributeImage Function" << endl;

    // Calculate dimensions of local image
    int rows = image.rows;
    int cols = image.cols;
    int localRows = rows / comm_sz;
    int localSize = localRows * cols;

    // Allocate memory for local image
    localImage.create(localRows, cols, CV_8UC1);

    // Scatter image data among processes
    MPI_Scatter(image.data, localSize, MPI_UNSIGNED_CHAR, localImage.data, localSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    cout << "Image data distributed among processes." << endl;
}

void calculateLocalHistogram(const Mat& image, vector<int>& localHist) {
    localHist.assign(256, 0);

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            int pixel = image.at<uchar>(i, j);
            localHist[pixel]++;
        }
    }
}

void combineHistograms(const vector<int>& localHist, vector<int>& globalHist) {
    MPI_Allreduce(localHist.data(), globalHist.data(), localHist.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}


void histogramEqualization(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    vector<int> localHist(256);
    vector<int> globalHist(256);
    vector<int> localEqualizedHist(256);

    // Load and convert the image to grayscale in rank 0
    if (my_rank == 0) {
        loadImage(imagePath, image);
        convertToGrayscale(image);
    }

    // Broadcast image dimensions
    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    // Calculate the number of pixels each process will handle
    int localPixels = (rows * cols) / comm_sz;
    // Create a local image buffer for each process
    Mat localImage(rows, cols, CV_8UC1);

    // Scatter the image data to all processes
    MPI_Scatter(image.data, localPixels, MPI_UNSIGNED_CHAR,
        localImage.data, localPixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double start_time, end_time;

    // Record start time
    start_time = MPI_Wtime();

    // Calculate the local histogram of the scattered image part
    calculateLocalHistogram(localImage, localHist);

    // Combine local histograms from all processes to obtain the global histogram
    combineHistograms(localHist, globalHist);

    // Calculate the cumulative distribution function (CDF) of the global histogram
    int totalPixels = rows * cols;
    for (int i = 0; i < 256; ++i) {
        float cumulativeSum = 0;
        for (int j = 0; j <= i; ++j) {
            cumulativeSum += globalHist[j];
        }
        // Normalize cumulative sum and map it to the range [0, 255]
        localEqualizedHist[i] = static_cast<int>((cumulativeSum / totalPixels) * 255);
    }

    // Apply histogram equalization to each pixel in the local image.
    for (int i = 0; i < localImage.rows; ++i) {
        for (int j = 0; j < localImage.cols; ++j) {
            int pixel = localImage.at<uchar>(i, j);
            localImage.at<uchar>(i, j) = localEqualizedHist[pixel];
        }
    }

    // Record end time
    end_time = MPI_Wtime();

    // Calculate execution time
    double execution_time = end_time - start_time;

    //cout << "Process " << my_rank << " is running... " << endl;

    // Show the equalized part of the image on each process
    /*imshow("Equalized Image Process " + to_string(my_rank), localImage);
    waitKey(0);*/

    // Gather equalized images
    Mat equalizedImage(image.size(), CV_8UC1);
    MPI_Gather(localImage.data, localPixels, MPI_UNSIGNED_CHAR, 
        equalizedImage.data, localPixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    // Display the equalized image
    if (my_rank == 0) {
        imshow("Original Image", image);
        cout << "Process " << my_rank << " displayed the images." << endl;
        imshow("Equalized Image", equalizedImage);
        cout << "Execution time: " << execution_time << " seconds" << endl;
        waitKey(0);
    }
}

void histogramEqualization3channels(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    vector<Mat> localChannels(3);
    vector<vector<int>> localHists(3, vector<int>(256));
    vector<vector<int>> globalHists(3, vector<int>(256));
    vector<vector<int>> localEqualizedHists(3, vector<int>(256));

    // Load the image in rank 0
    if (my_rank == 0) {
        image =  imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Unable to read the image " << imagePath << endl;
            return;
        }
    }

    // Broadcast image dimensions
    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    // Calculate the number of pixels each process will handle
    int localPixels = (rows * cols) / comm_sz;

    // Separate color channels and scatter the data to all processes
    vector< Mat> channels;
     split(image, channels);
    for (int c = 0; c < 3; ++c) {
        localChannels[c] = channels[c].clone();
        MPI_Scatter(channels[c].data, localPixels, MPI_UNSIGNED_CHAR,
            localChannels[c].data, localPixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    // Calculate local histograms for each color channel
    for (int c = 0; c < 3; ++c) {
        calculateLocalHistogram(localChannels[c], localHists[c]);
    }

    // Combine local histograms from all processes to obtain the global histogram for each channel
    for (int c = 0; c < 3; ++c) {
        combineHistograms(localHists[c], globalHists[c]);
    }

    // Calculate the cumulative distribution function (CDF) for each color channel
    int totalPixels = rows * cols;
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 256; ++i) {
            float cumulativeSum = 0;
            for (int j = 0; j <= i; ++j) {
                cumulativeSum += globalHists[c][j];
            }
            // Normalize cumulative sum and map it to the range [0, 255]
            localEqualizedHists[c][i] = static_cast<int>((cumulativeSum / totalPixels) * 255);
        }
    }

    // Apply histogram equalization to each pixel in each color channel
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < localChannels[c].rows; ++i) {
            for (int j = 0; j < localChannels[c].cols; ++j) {
                int pixel = localChannels[c].at<uchar>(i, j);
                localChannels[c].at<uchar>(i, j) = localEqualizedHists[c][pixel];
            }
        }
    }

    cout << "Process " << my_rank << " is running... " << endl;

    // Show the equalized part of the image on each process (for testing)
    for (int c = 0; c < 3; ++c) {
         imshow("Equalized Image Process " + to_string(my_rank) + " Channel " + to_string(c), localChannels[c]);
    }
     waitKey(0);

    // Gather equalized images
    vector< Mat> equalizedChannels(3,  Mat(rows, cols, CV_8UC1));
    for (int c = 0; c < 3; ++c) {
        MPI_Gather(localChannels[c].data, localPixels, MPI_UNSIGNED_CHAR, equalizedChannels[c].data, localPixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    }

    // Display the equalized image
    if (my_rank == 0) {
         Mat equalizedImage;
         merge(equalizedChannels, equalizedImage);
         imshow("Original Image", image);
        cout << "Process " << my_rank << " displayed the images." << endl;
         imshow("Equalized Image", equalizedImage);
         waitKey(0);
    }
}

void applyGaussianBlur(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    int kernelSize = 5;
    // Load and convert the image to grayscale in rank 0
    if (my_rank == 0) {
        loadImage(imagePath, image);
        convertToGrayscale(image);
    }

    // Broadcast image dimensions
    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    // Calculate the number of pixels each process will handle
    int localPixels = (rows * cols) / comm_sz;

    // Create a local image buffer for each process
    Mat localImage(rows, cols, CV_8UC1);

    // Scatter the image data to all processes
    MPI_Scatter(image.data, localPixels, MPI_UNSIGNED_CHAR, localImage.data, localPixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double start_time, end_time;

    // Record start time
    start_time = MPI_Wtime();

    // Apply Gaussian blur to the scattered image part
    GaussianBlur(localImage, localImage, Size(kernelSize, kernelSize), 0, 0, BORDER_DEFAULT);

    // Record end time
    end_time = MPI_Wtime();

    // Calculate execution time
    double execution_time = end_time - start_time;

    //cout << "Process " << my_rank << " is running... " << endl;

    //// Show the blurred part of the image on each process
    //imshow("Blurred Image Process " + to_string(my_rank), localImage);
    //waitKey(0);

    // Gather blurred images
    Mat blurredImage(rows, cols, CV_8UC1); // Allocate the receiving buffer
    MPI_Gather(localImage.data, localPixels, MPI_UNSIGNED_CHAR, blurredImage.data, localPixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Display the blurred image (only in rank 0)
    if (my_rank == 0) {
        imshow("Blurred Image", blurredImage);
        cout << "Process " << my_rank << " displayed the images." << endl;
        imshow("Original Image", image);
        cout << "Execution time: " << execution_time << " seconds" << endl;
        waitKey(0);
    }
}

void edgeDetection(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    if (my_rank == 0) {
        loadImage(imagePath, image);
    }

    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    int localPixels = (rows * cols) / comm_sz;
    Mat localImage(rows, cols, CV_8UC3);
    Mat edgesLocalImage(rows, cols, CV_8UC1);

    MPI_Scatter(image.data, localPixels * 3, MPI_UNSIGNED_CHAR, localImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double start_time, end_time;

    // Record start time
    start_time = MPI_Wtime();

    // edge detection on localImage
    cvtColor(localImage, localImage, COLOR_BGR2GRAY); // Convert to grayscale
    Canny(localImage, edgesLocalImage, 100, 200); // Apply Canny edge detection

    // Record end time
    end_time = MPI_Wtime();

    // Calculate execution time
    double execution_time = end_time - start_time;

    //// Display the edge-detected image
    //imshow("Edges Image", edgesLocalImage);
    //waitKey(0);

    Mat coloredEdgesImage(rows, cols, CV_8UC1);
    MPI_Gather(edgesLocalImage.data, localPixels, MPI_UNSIGNED_CHAR, coloredEdgesImage.data, localPixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        imshow("Colored Edges Image", coloredEdgesImage);
        cout << "Process " << my_rank << " displayed the edge-detected images." << endl;
        imshow("Original Image", image);
        cout << "Execution time of Gaussian blur: " << execution_time << " seconds" << endl;
        waitKey(0);
    }
}

void imageRotation(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    if (my_rank == 0) {
        loadImage(imagePath, image);
    }

    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    int localPixels = (rows * cols) / comm_sz;
    Mat localImage(rows, cols, CV_8UC3);
    Mat rotatedLocalImage(rows, cols, CV_8UC3);

    MPI_Scatter(image.data, localPixels * 3, MPI_UNSIGNED_CHAR, localImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Perform image rotation on localImage
    rotate(localImage, rotatedLocalImage, ROTATE_90_CLOCKWISE);

    // Display the rotated image
    imshow("Rotated Image", rotatedLocalImage);
    waitKey(0);

    Mat finalImage(cols, rows, CV_8UC3);
    MPI_Gather(rotatedLocalImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, finalImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        imshow("Final Image", finalImage);
        cout << "Process " << my_rank << " displayed the images." << endl;
        imshow("Original Image", image);
        waitKey(0);
    }
}

//void imageRotation(const string& imagePath, int my_rank, int comm_sz) {
//    Mat image; 
//
//    // Load the image on rank 0
//    if (my_rank == 0) {
//        image = imread(imagePath, IMREAD_COLOR);
//    }
//
//    // Broadcast image dimensions
//    int rows, cols;
//    if (my_rank == 0) {
//        rows = image.rows;
//        cols = image.cols;
//    }
//    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//    // Calculate the number of rows each process will handle
//    int localRows = rows / comm_sz;
//
//    // Create a local image buffer for each process
//    Mat localImage(localRows, cols, CV_8UC3);
//
//    // Scatter image data to all processes
//    MPI_Scatter(image.data, localRows * cols * 3, MPI_UNSIGNED_CHAR, localImage.data, localRows * cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
//
//    // Perform image rotation locally
//    Mat localRotated;
//    Point2f center(cols / 2.0, localRows / 2.0); // Calculate the center of the image
//    double angle = 45.0; // Rotation angle in degrees
//    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0); // Create the rotation matrix
//    warpAffine(localImage, localRotated, rotationMatrix, Size(cols, localRows)); // Apply the rotation
//
//    cout << "Process " << my_rank << " is running... " << endl;
//
//    // Show the rotated part of the image on each process
//    imshow("Rotated Image Process " + to_string(my_rank), localRotated);
//    waitKey(0);
//
//    // Gather rotated images
//    Mat gatheredRotated(rows, cols, CV_8UC3); // Allocate the receiving buffer
//    MPI_Gather(localRotated.data, localRows * cols * 3, MPI_UNSIGNED_CHAR, gatheredRotated.data, localRows * cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
//
//    // Display the final rotated image (only in rank 0)
//    if (my_rank == 0) {
//        // Combine the gathered rotated images to reconstruct the final rotated image
//        imshow("Rotated Image", gatheredRotated);
//        cout << "Process " << my_rank << " displayed the final rotated image." << endl;
//        imshow("Original Image", image);
//        waitKey(0);
//    }
//}

void imageScaling(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    if (my_rank == 0) {
        loadImage(imagePath, image);
    }

    // Broadcast image dimensions
    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    // Calculate the number of rows each process will handle
    int localRows = rows / comm_sz;

    // Create a local image buffer for each process
    Mat localImage(localRows, cols, CV_8UC3);

    // Scatter image data to all processes
    MPI_Scatter(image.data, localRows * cols * 3, MPI_UNSIGNED_CHAR, localImage.data, localRows * cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double start_time, end_time;

    // Record start time
    start_time = MPI_Wtime();

    // Perform image scaling locally
    Mat localScaled;
    resize(localImage, localScaled, Size(cols * 2, localRows * 2), 0, 0, INTER_LINEAR); // Scale the image to double its size

    end_time = MPI_Wtime();
    // Calculate execution time
    double execution_time = end_time - start_time;

    //cout << "Process " << my_rank << " is running... " << endl;

    //// Show the scaled part of the image on each process
    //imshow("Scaled Image Process " + to_string(my_rank), localScaled);
    //waitKey(0);

    // Gather scaled images
    Mat gatheredScaled(rows * 2, cols * 2, CV_8UC3); // Allocate the receiving buffer
    MPI_Gather(localScaled.data, localRows * cols * 3 * 4, MPI_UNSIGNED_CHAR, gatheredScaled.data, localRows * cols * 3 * 4, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Display the scaled image (only in rank 0)
    if (my_rank == 0) {
        imshow("Scaled Image", gatheredScaled);
        cout << "Process " << my_rank << " displayed the images." << endl;
        imshow("Original Image", image);
        cout << "Execution time of Gaussian blur: " << execution_time << " seconds" << endl;
        waitKey(0);
    }
}

void colorSpaceConversion(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    if (my_rank == 0) {
        loadImage(imagePath, image);
    }

    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    int localPixels = (rows * cols) / comm_sz;
    Mat localImage(rows, cols, CV_8UC3);

    MPI_Scatter(image.data, localPixels * 3, MPI_UNSIGNED_CHAR, localImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double start_time, end_time;

    // Record start time
    start_time = MPI_Wtime();

    cvtColor(localImage, localImage, COLOR_BGR2HSV);

    // Record end time
    end_time = MPI_Wtime();

    // Calculate execution time
    double execution_time = end_time - start_time;

    //imshow("HSV Image", localImage);
    //waitKey(0);
    ////destroyAllWindows();

    Mat coloredImage(rows, cols, CV_8UC3);
    MPI_Gather(localImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, coloredImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        imshow("Colored Image", coloredImage);
        cout << "Process " << my_rank << " displayed the images." << endl;
        imshow("Original Image", image);
        cout << "Execution time of color Space Conversion: " << execution_time << " seconds" << endl;
        waitKey(0);
    }
}

void globalThresholding(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    if (my_rank == 0) {
        loadImage(imagePath, image);
    }

    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    int localPixels = (rows * cols) / comm_sz;
    Mat localImage(rows, cols, CV_8UC3);
    Mat thresholdedLocalImage(rows, cols, CV_8UC1);

    MPI_Scatter(image.data, localPixels * 3, MPI_UNSIGNED_CHAR, localImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    double start_time, end_time;

    // Record start time
    start_time = MPI_Wtime();

    // Perform global thresholding on localImage
    int thresholdValue = 128; // Set your threshold value here
    int maxValue = 255; // Maximum pixel value after thresholding
    threshold(localImage, thresholdedLocalImage, thresholdValue, maxValue, THRESH_BINARY);

    end_time = MPI_Wtime();
    double execution_time = end_time - start_time;

    //// Display the thresholded image
    //imshow("Thresholded Image", thresholdedLocalImage);
    //waitKey(0);

    Mat coloredImage(rows, cols, CV_8UC3);
    MPI_Gather(thresholdedLocalImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, coloredImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        imshow("Colored Image", coloredImage);
        cout << "Process " << my_rank << " displayed the images." << endl;
        imshow("Original Image", image);
        cout << "Execution time of global Thresholding: " << execution_time << " seconds" << endl;
        waitKey(0);
    }
}

void localThresholding(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;

    if (my_rank == 0) {
        loadImage(imagePath, image);
    }

    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    int localPixels = (rows * cols) / comm_sz;
    Mat localImage(rows, cols, CV_8UC3);
    Mat thresholdedLocalImage(rows, cols, CV_8UC1);

    MPI_Scatter(image.data, localPixels * 3, MPI_UNSIGNED_CHAR, localImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Perform local thresholding on localImage
    cvtColor(localImage, thresholdedLocalImage, COLOR_BGR2GRAY);
    threshold(thresholdedLocalImage, thresholdedLocalImage, 128, 255, THRESH_BINARY);

    // Display the thresholded image
    imshow("Thresholded Image", thresholdedLocalImage);
    waitKey(0);

    Mat coloredImage(rows, cols, CV_8UC1);
    MPI_Gather(thresholdedLocalImage.data, localPixels, MPI_UNSIGNED_CHAR, coloredImage.data, localPixels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        imshow("Colored Image", coloredImage);
        cout << "Process " << my_rank << " displayed the images." << endl;
        imshow("Original Image", image);
        waitKey(0);
    }
}

void compressImage(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    if (my_rank == 0) {
        loadImage(imagePath, image);
    }

    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    int localPixels = (rows * cols) / comm_sz;
    Mat localImage(rows, cols, CV_8UC3);
    Mat compressedLocalImage;

    MPI_Scatter(image.data, localPixels * 3, MPI_UNSIGNED_CHAR, localImage.data, localPixels * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Compress the local image using JPEG compression
    vector<uchar> compressedData;
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(70); // Set JPEG quality level (0-100)
    imencode(".jpg", localImage, compressedData, compression_params);
    imwrite(to_string(my_rank) + "compressed_image.jpg", localImage);

    // Gather the compressed data back to process 0
    vector<uchar> allCompressedData;
    if (my_rank == 0) {
        allCompressedData.resize(comm_sz * localPixels * 3); // Increase buffer size
    }
    MPI_Gather(compressedData.data(), compressedData.size(), MPI_UNSIGNED_CHAR, allCompressedData.data(), compressedData.size(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        Mat finalImage = imdecode(allCompressedData, IMREAD_COLOR);
        imwrite("compressed_image.jpg", finalImage);
        cout << "Image compressed and saved as compressed_image.jpg" << endl;
    }
}

void median(const string& imagePath, int my_rank, int comm_sz) {
    Mat image;
    int kernelSize = 5;
    if (my_rank == 0) {
        loadImage(imagePath, image);
    }

    // Broadcast image dimensions
    int rows, cols;
    broadcastImageDimensions(image, rows, cols, my_rank);

    // Calculate the number of rows each process will handle
    int localRows = rows / comm_sz;

    // Create a local image buffer for each process
    Mat localImage(localRows, cols, CV_8UC1);

    // Scatter image data to all processes
    MPI_Scatter(image.data, localRows * cols, MPI_UNSIGNED_CHAR, localImage.data, localRows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Apply median filter locally
    Mat localFiltered;
    medianBlur(localImage, localFiltered, kernelSize); // Apply median filter with specified kernel size

    cout << "Process " << my_rank << " is running... " << endl;

    // Show the filtered part of the image on each process
    imshow("Filtered Image Process " + to_string(my_rank), localFiltered);
    waitKey(0);

    // Gather filtered images
    Mat gatheredFiltered(rows, cols, CV_8UC1); // Allocate the receiving buffer
    MPI_Gather(localFiltered.data, localRows * cols, MPI_UNSIGNED_CHAR, gatheredFiltered.data, localRows * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Display the filtered image (only in rank 0)
    if (my_rank == 0) {
        imshow("Filtered Image", gatheredFiltered);
        cout << "Process " << my_rank << " displayed the images." << endl;
        imshow("Original Image", image);
        waitKey(0);
    }
}