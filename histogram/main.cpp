#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {

    Mat image;
    image = cv::imread("C:\\Users\\divya\\Desktop\\cpp\\opencv\\lena.jpg");

    vector<Mat> channels;

    split(image, channels);

	int hsize = 256;

    float range[] = { 0, 256 }; 
	const float* histranges[] = { range };

    bool uniform = true, accumulate = false;

    Mat b, g, r;
    calcHist( &channels[0], 1, 0, Mat(), b, 1,  &hsize, histranges, uniform, accumulate );

    calcHist( &channels[1], 1, 0, Mat(), g, 1,  &hsize, histranges, uniform, accumulate );

    calcHist( &channels[2], 1, 0, Mat(), r, 1,  &hsize, histranges, uniform, accumulate );

    int hist_w= 512, hist_h=400;
    int bin_w = cvRound( (double) hist_w/hsize );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    normalize(b, b, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g, g, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r, r, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    for( int i = 1; i < hsize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(b.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(g.at<float>(i)) ),
              Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(r.at<float>(i)) ),
              Scalar( 0, 0, 255), 2, 8, 0  );
    }

    // luma  
    Mat bl = channels[0].clone();
    Mat gl = channels[1].clone();
    Mat rl = channels[2].clone();

    int img_h =image.rows;
    int img_w = image.cols;

    Mat luma = Mat::zeros(img_h, img_w, CV_8UC1);

    for(int i =0; i<img_h;  i++){
        for(int j=0; j<img_w;  j++){
            uint8_t l = rl.at<uint8_t>(i, j)*0.2126 + gl.at<uint8_t>(i, j)*0.7152 + bl.at<uint8_t>(i, j)*0.0722;

            luma.at<uint8_t>(i,j)=l;
        }
    }

    Mat lumahist;
    calcHist( &luma, 1, 0, Mat(), lumahist, 1,  &hsize, histranges, uniform, accumulate );

    Mat lumahistimg( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    for( int i = 1; i < hsize; i++ ){
        line( lumahistimg, Point( bin_w*(i-1), hist_h - cvRound(lumahist.at<float>(i-1)) ), Point( bin_w*(i), hist_h - cvRound(lumahist.at<float>(i)) ), Scalar( 255, 0, 0), 2, 8, 0  );
    }
    imshow("Source image", image );
    imshow("calcHist", histImage);
    imshow("luma histogram", lumahistimg);
    imwrite("C:\\Users\\divya\\Desktop\\cpp\\opencv\\calchist.jpg",histImage);
    imwrite("C:\\Users\\divya\\Desktop\\cpp\\opencv\\luma_hist.jpg",lumahistimg);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
