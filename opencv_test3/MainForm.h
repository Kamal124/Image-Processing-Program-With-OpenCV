#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>  // we need add one more lib file imgproc
#include "jackylib.h" 
#include <fstream>
#include <vector>
#include <bitset>

cv::Mat output_mat , output_mat2;
cv::Mat target_mat;
int slidervalue, sigma_glo = 0, threshold_gol= 154;
double slidervalue_rot, scale_gol;
cv::Mat public_matrix;
bool over_write_flag1, over_write_flag2,over_write_flag3, over_write_flag_flt, over_write_flag_seg;

namespace opencv_test3 {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace std;
	using namespace cv;
	using namespace jacky_lib;
	 
 cv::Mat smoothing_filter(Mat input)
	{

		Mat im, disp;
		cvtColor(input, im, CV_BGR2GRAY);
		Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
		for (int r = 1; r < im.rows - 1; r++)
		{
			for (int c = 1; c < im.cols - 1; c++)
			{
				output.at<uchar>(r, c) = 1.0 / 9.0 *
					(im.at<uchar>(r, c)
						+ im.at<uchar>(r - 1, c - 1)
						+ im.at<uchar>(r - 1, c)
						+ im.at<uchar>(r - 1, c + 1)
						+ im.at<uchar>(r, c - 1)
						+ im.at<uchar>(r, c + 1)
						+ im.at<uchar>(r + 1, c - 1)
						+ im.at<uchar>(r + 1, c)
						+ im.at<uchar>(r + 1, c + 1));
			}
		}
		cvtColor(output, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp);
		return disp;
	}

 Mat prewitt_sharpening(Mat input)
 {
	 Mat im, disp;
	 cvtColor(input, im, CV_BGR2GRAY);
	 Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
	 Mat ph_output(im.rows, im.cols, CV_8UC1, Scalar(0));
	 Mat pv_output(im.rows, im.cols, CV_8UC1, Scalar(0));
	 float p_mag;
	 int mag;
	 for (int r = 1; r < im.rows - 1; r++)
	 {
		 for (int c = 1; c < im.cols - 1; c++)
		 {
			 ph_output.at<uchar>(r, c) =
				 abs(im.at<uchar>(r, c) * 0
					 + im.at<uchar>(r - 1, c - 1) * 1
					 + im.at<uchar>(r - 1, c) * 1
					 + im.at<uchar>(r - 1, c + 1) * 1
					 + im.at<uchar>(r, c - 1) * 0
					 + im.at<uchar>(r, c + 1) * 0
					 + im.at<uchar>(r + 1, c - 1) * (-1)
					 + im.at<uchar>(r + 1, c) * (-1)
					 + im.at<uchar>(r + 1, c + 1) * (-1));

			 pv_output.at<uchar>(r, c) =
				 abs(im.at<uchar>(r, c) * 0
					 + im.at<uchar>(r - 1, c - 1) * 1
					 + im.at<uchar>(r - 1, c) * 0
					 + im.at<uchar>(r - 1, c + 1) * (-1)
					 + im.at<uchar>(r, c - 1) * 1
					 + im.at<uchar>(r, c + 1) * (-1)
					 + im.at<uchar>(r + 1, c - 1) * 1
					 + im.at<uchar>(r + 1, c) * 0
					 + im.at<uchar>(r + 1, c + 1) * (-1));

			 p_mag = (ph_output.at<uchar>(r, c) * ph_output.at<uchar>(r, c)) + (pv_output.at<uchar>(r, c) * pv_output.at<uchar>(r, c));
			 mag = sqrt(p_mag);
			 if (mag > 100)
			 {
				 output.at<uchar>(r, c) = 255;
			 }
			 else
			 {
				 output.at<uchar>(r, c) = 0;
			 }
		 }
	 }
	 cvtColor(output, disp, CV_GRAY2BGR);
	 convertScaleAbs(disp, disp);
	 return disp;
 }

 Mat Bright_Inc(Mat input)
 {
	 Mat im, disp;
	 cvtColor(input, im, CV_BGR2GRAY);
	 Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
	 for (int r = 0; r < im.rows; r++)
	 {
		 for (int c = 0; c < im.cols; c++)
		 {
			 if (output.at<uchar>(r, c) = im.at<uchar>(r, c) + slidervalue > 255)
				 output.at<uchar>(r, c) = 255;
			 else
			 {
				 output.at<uchar>(r, c) = im.at<uchar>(r, c) + slidervalue;
			 }
		 }
	 }
	 cvtColor(output, disp, CV_GRAY2BGR);
	 convertScaleAbs(disp, disp);
	 return disp;
 }

 Mat Bright_Dec(Mat input)
 {
	 Mat im, disp;
	 cvtColor(input, im, CV_BGR2GRAY);
	 Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
	 for (int r = 0; r < im.rows; r++)
	 {
		 for (int c = 0; c < im.cols; c++)
		 {
			 if (output.at<uchar>(r, c) = im.at<uchar>(r, c) - slidervalue*5 < 0)
				 output.at<uchar>(r, c) = 0;
			 else
			 {
				 output.at<uchar>(r, c) = im.at<uchar>(r, c) - slidervalue*5;
			 }
		 }
	 }
	 cvtColor(output, disp, CV_GRAY2BGR);
	 convertScaleAbs(disp, disp);
	 return disp;
 }

Mat Image_blinding(Mat input1, Mat input2)
 {
	 Mat image1, image2, disp;
	 cvtColor(input1, image1, CV_BGR2GRAY);
	 cvtColor(input2, image2, CV_BGR2GRAY);
	 resize(image2, image2, cv::Size(image1.cols, image1.rows));
	 Mat imageout(image1.rows, image1.cols, CV_8UC1);
	 for (int r = 0; r < image1.rows; r++)
	 {
		 for (int c = 0; c < image1.cols; c++)
		 {
			 //if ((imageout.at<uchar>(r, c) = image1.at<uchar>(r, c) * (0.3 +slidervalue/70)  + image2.at<uchar>(r, c) * (0.7 + slidervalue/60)) > 255)
			 if ((imageout.at<uchar>(r, c) = (image1.at<uchar>(r, c) * (0.4*( 100 /slidervalue  )) )+ (0.6*(image2.at<uchar>(r, c) * ( (slidervalue+100) / 100)))) > 255)
				 imageout.at<uchar>(r, c) = 255;
			 else if ((imageout.at<uchar>(r, c) = (image1.at<uchar>(r, c) * (0.4*(100 / slidervalue )) )+ (0.6*(image2.at<uchar>(r, c) * ((slidervalue +100) / 100)))) < 0)
				 imageout.at<uchar>(r, c) = 0;
			 else
			 {
				 //imageout.at<uchar>(r, c) = image1.at<uchar>(r, c) * (0.3 + slidervalue / 70) + image2.at<uchar>(r, c) * (0.7 + slidervalue / 60);
				 imageout.at<uchar>(r, c) = (image1.at<uchar>(r, c) * (0.4*(100 / slidervalue ))) + (0.6*(image2.at<uchar>(r, c) * ((slidervalue+100) / 100)));
			 }
		 }
	 }
	 cvtColor(imageout, disp, CV_GRAY2BGR);
	 convertScaleAbs(disp, disp);
	 return disp;
 }

Mat Log_Transformation(Mat input, int log_var)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
	for (int r = 0; r < im.rows; r++)
	{
		for (int c = 0; c < im.cols; c++)
		{
			output.at<uchar>(r, c) = log_var * (log10f(1 + (im.at<uchar>(r, c))));
		}
	}
	cvtColor(output, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

Mat Power_Transformation(Mat input, int gamma)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
	for (int r = 0; r < im.rows; r++)
	{
		for (int c = 0; c < im.cols; c++)
		{
			output.at<uchar>(r, c) = powf(im.at<uchar>(r, c), gamma/10);
		}
	}
	cvtColor(output, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

Mat Negative(Mat input)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
	for (int r = 0; r < im.rows; r++)
	{
		for (int c = 0; c < im.cols; c++)
		{
			output.at<uchar>(r, c) = 255 - 1 - (im.at<uchar>(r, c));
		}
	}
	//imshow(output, 0);
	cvtColor(output, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

Mat histogram(Mat input)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
	float hist[256] = { 0 };
	float hist2[256] = { 0 };
	float prob[256];
	float comm[256];
	/////////////////////////////////histogram//////////////////////////// 
	for (int g = 0; g < 256; g++)
	{
		for (int r = 0; r < im.rows; r++)
		{
			for (int c = 0; c < im.cols; c++)
			{
				if (im.at<uchar>(r, c) == g)
					hist[g] = hist[g] + 1;
			}
		}
	}
	///////////////////prob//////////////////////////////// 
	for (int j = 0; j < 256; j++)
	{
		prob[j] = hist[j] / im.total();
	}
	///////////////////commulative////////// 
	comm[0] = prob[0];
	for (int j = 1; j < 256; j++)
	{
		comm[j] = comm[j - 1] + prob[j];
	}
	////////////////////////////////map to image//////////////////////// 
	for (int r = 0; r < im.rows; r++)
	{
		for (int c = 0; c < im.cols; c++)
		{
			output.at<uchar>(r, c) = comm[im.at<uchar>(r, c)] * 255.0;
		}
	}
	cvtColor(output, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

Mat median_filter(Mat img_name)
{
	int temp;
	//Mat input = imread(img_name, 0);
	Mat input, disp;
	cvtColor(img_name, input, CV_BGR2GRAY);
	Mat outputm(input.rows, input.cols, CV_8UC1);
	////////////////////////median filter////////////
	int f[9];
	for (int r = 1; r < input.rows - 1; r++)
		for (int c = 1; c < input.cols - 1; c++)
		{
			f[0] = input.at<uchar>(r - 1, c - 1);
			f[1] = input.at<uchar>(r - 1, c);
			f[2] = input.at<uchar>(r - 1, c + 1);
			f[3] = input.at<uchar>(r, c - 1);
			f[4] = input.at<uchar>(r, c);
			f[5] = input.at<uchar>(r, c + 1);
			f[6] = input.at<uchar>(r + 1, c - 1);
			f[7] = input.at<uchar>(r + 1, c);
			f[8] = input.at<uchar>(r + 1, c + 1);
			sort(f, f + 9);
			outputm.at<uchar>(r, c) = f[4];
		}
	/*namedWindow("Median filter", 0);
	imshow("Median filter", outputm);
	waitKey(0);*/
	cvtColor(outputm, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

Mat image_flipping(Mat input)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat output(im.rows, im.cols, CV_8UC1, Scalar(0));
	for (int r = 0; r < im.rows; r++)
	{
		int cf = im.cols - 1;
		for (int c = 0; c < im.cols; c++)
		{
			output.at<uchar>(r, c) = im.at<uchar>(r, cf);
			cf--;
		}
	}
	cvtColor(output, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

int Round(float in)
{
	int x = (int)in;
	if ((in - x) >= 0.5)
		return (x + 1);
	else
		return x;

}

Mat zooming(Mat input)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat crop(im.rows * 0.5, im.cols * 0.5, CV_8UC1);
	Mat zoom(im.rows, im.cols, CV_8UC1);
	int rc = im.rows / 4;
	for (int r = 0; r < crop.rows; r++)
	{
		int cc = im.cols / 4;
		for (int c = 0; c < crop.cols; c++)
		{
			crop.at<uchar>(r, c) = im.at<uchar>(rc, cc);
			cc++;
		}
		rc++;
	}
	float ri = crop.rows;
	float ci = crop.cols;
	float rz = zoom.rows;
	float cz = zoom.cols;
	float sc = (ci - 1) / cz;
	float sr = (ri - 1) / rz;
	for (int r = 0; r < zoom.rows; r++)
	{
		for (int c = 0; c < zoom.cols; c++)
		{
			float rzz = sr * r;
			float czz = sc * c;
			int x = Round(rzz);
			int y = Round(czz);
			zoom.at<uchar>(r, c) = crop.at<uchar>(x, y);
		}
	}
	cvtColor(zoom, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

Mat rotation90(Mat input)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat ImR90(input.cols, input.rows, CV_8UC1, Scalar(0));
	//image after R90  
	/*Mat ImR270(input.cols, input.rows, CV_8UC1);
	//image after R270  
	Mat ImR180(input.rows, input.cols, CV_8UC1);
	//image after R180  namedWindow("imageR", 0);*/  
	int c90 = 0;  
	for (int r = 0; r < ImR90.rows; r++)  
	{   
		int r90 = im.rows - 1;   
	for (int c = 0; c < ImR90.cols; c++)   
	{    
		ImR90.at<uchar>(r, c) = im.at<uchar>(r90, c90);    
		r90--;   
	}   
	c90++;  
	} 
	cvtColor(ImR90, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}
Mat rotation180(Mat input) {
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat ImR180(im.rows, im.cols, CV_8UC1, Scalar(0));
	flip(im, ImR180, -1);
	cvtColor(ImR180, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}
Mat rotation270(Mat input) 
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat ImR270(im.cols, im.rows, CV_8UC1, Scalar(0));
	transpose(im, ImR270);
	flip(ImR270, ImR270, 0);
	cvtColor(ImR270, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

Mat rotat_any_ang(Mat input,double ang, double scale)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat output(im.cols, im.rows, CV_8UC1, Scalar(0));
	/*int m = getOptimalDFTSize(output.rows);
	int n = getOptimalDFTSize(output.cols);
	copyMakeBorder(output, disp, 0, m - im.rows, 0, n - im.cols, 0, Scalar(0));*/
	//wrapAffine();
	//output = rotat_any_ang(im,ang);
	Point2f pt(im.rows / 2, im.cols / 2);
	//output = getRotationMatrix2D(pt, ang, scale);
	Mat r = getRotationMatrix2D(pt, ang, scale);
	warpAffine(im, output, r, cv::Size(im.cols, im.rows));
	cvtColor(output, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

Mat Frequency_Domain_Smoothing_Filter(Mat input,int returned_one, int sigma )
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat output_ret(im.rows, im.cols, CV_8UC1, Scalar(0));
	Mat padded, padded1, mag, mag2, ideal_low_pass_ret;
	
		//	1.	Expand the image to an optimal size
		int m = getOptimalDFTSize(im.rows);
		int n = getOptimalDFTSize(im.cols);
		copyMakeBorder(im, padded, 0, m - im.rows, 0, n - im.cols, 0, Scalar(0));
		//padded1 = padded;
		//imshow("padded", padded);//************************************************************************************1

		//	2.	Make place for both the complex and the real values.
		padded.convertTo(padded, CV_32FC1, 1.0 / 255.0);
		Mat output[] = { padded,Mat::zeros(padded.size(),CV_32FC1) };
		Mat complexI;
		merge(output, 2, complexI);

		//	3.	make the Discrete Fourier Transform.
		dft(complexI, complexI);

		//	4.	Transform the real and complex values to magnitude for visualization. 
		split(complexI, output);
		magnitude(output[0], output[1], mag);
		//mag1 = mag;
		//normalize(mag, mag, 0, 1, CV_MINMAX);
		//imshow("before swap", mag);//***********************************************************************************2

//	5.	Crop and rearrange.
		int cx = output[0].cols / 2;
		int cy = output[0].rows / 2;
		Mat p1_r(output[0], Rect(0, 0, cx, cy));
		Mat p2_r(output[0], Rect(cx, 0, cx, cy));
		Mat p3_r(output[0], Rect(0, cy, cx, cy));
		Mat p4_r(output[0], Rect(cx, cy, cx, cy));

		Mat temp;
		p1_r.copyTo(temp);
		p4_r.copyTo(p1_r);
		temp.copyTo(p4_r);
		p2_r.copyTo(temp);
		p3_r.copyTo(p2_r);
		temp.copyTo(p3_r);

		Mat p1_i(output[1], Rect(0, 0, cx, cy));
		Mat p2_i(output[1], Rect(cx, 0, cx, cy));
		Mat p3_i(output[1], Rect(0, cy, cx, cy));
		Mat p4_i(output[1], Rect(cx, cy, cx, cy));

		p1_i.copyTo(temp);
		p4_i.copyTo(p1_i);
		temp.copyTo(p4_i);
		p2_i.copyTo(temp);
		p3_i.copyTo(p2_i);
		temp.copyTo(p3_i);

		magnitude(output[0], output[1], mag2);
		//normalize(mag, mag, 1, 0, CV_MINMAX);
		//imshow("after swap", mag);//******************************************************************************************3
		//namedWindow("after IDFT", 0);
		//mag2 = mag;

		//	6.	apply filter 

		//createTrackbar("sigma", "after IDFT", &sigma, 100);
		// make filter first 
		Mat ideal_low_pass(padded.size(), CV_32FC1);
		float d0 = sigma;//The smaller the radius D0, the larger the blur; the larger the radius D0, the smaller the blur
		for (int i = 0; i < padded.rows; i++)
		{
			for (int j = 0; j < padded.cols; j++)
			{
				double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));//Molecule, Calculate pow must be float type
				if (d <= d0)
				{
					ideal_low_pass.at<float>(i, j) = 1;
				}
				else
				{
					ideal_low_pass.at<float>(i, j) = 0;
				}
			}
		}
		ideal_low_pass_ret = ideal_low_pass;
		//imshow("ideal low pass", ideal_low_pass);//*****************************************************************************************4
		//then apply filter on both real and imaginary parts.
		Mat blur_r, blur_i, complexF;
		multiply(output[0], ideal_low_pass, blur_r); //filter (the real part is multiplied by the corresponding element of the filter template)
		multiply(output[1], ideal_low_pass, blur_i); // filter (imaginary part is multiplied by the corresponding element of the filter template)
		Mat plane1[] = { blur_r, blur_i };
		merge(plane1, 2, complexF);//The real and imaginary parts merge
//	7.	Take inverse
		idft(complexF, complexF); //idft result is also plural
		split(complexF, output);
		magnitude(output[0], output[1], output[0]);

		//	8.normalized for easy display

		normalize(output[0], output_ret, 1, 0, CV_MINMAX);
		//output_ret = output[0];
		//imshow("after IDFT", output[0]);//****************************************************************************************************5
		//waitKey(0);
	//return padded;-------------1
	//return mag;----------------2
	//return mag2;---------------3
	//return ideal_low_pass;-----4
	//return output[0];----------5
	if (returned_one == 1)
	{
		/*cvtColor(padded1, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		CV_32FC1*/
		/*padded.convertTo(padded, CV_8UC1);
		cvtColor(padded, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255);
		return disp;*/
		return padded;
	}
	else if (returned_one == 2)
	{
		cvtColor(mag, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		//return mag;
	}
	else if (returned_one == 3)
	{
		cvtColor(mag2, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		//return mag2;
	}
	else if (returned_one == 4)
	{
		cvtColor(ideal_low_pass_ret, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		//return ideal_low_pass_ret;
	}
	else if (returned_one == 5)
	{
		cvtColor(output_ret, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp,255,0);
		return disp;
		//return output_ret;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat Frequency_Domain_Sharpening_Filter(Mat input, int returned_one, int sigma)
{
	Mat im, disp;
	cvtColor(input, im, CV_BGR2GRAY);
	Mat output_ret(im.rows, im.cols, CV_8UC1, Scalar(0));
	Mat padded, padded1, mag, mag2, ideal_low_pass_ret;

	//	1.	Expand the image to an optimal size
	int m = getOptimalDFTSize(im.rows);
	int n = getOptimalDFTSize(im.cols);
	copyMakeBorder(im, padded, 0, m - im.rows, 0, n - im.cols, 0, Scalar(0));
	//padded1 = padded;
	//imshow("padded", padded);//************************************************************************************1

	//	2.	Make place for both the complex and the real values.
	padded.convertTo(padded, CV_32FC1, 1.0 / 255.0);
	Mat output[] = { padded,Mat::zeros(padded.size(),CV_32FC1) };
	Mat complexI;
	merge(output, 2, complexI);

	//	3.	make the Discrete Fourier Transform.
	dft(complexI, complexI);

	//	4.	Transform the real and complex values to magnitude for visualization. 
	split(complexI, output);
	magnitude(output[0], output[1], mag);
	//mag1 = mag;
	//normalize(mag, mag, 0, 1, CV_MINMAX);
	//imshow("before swap", mag);//***********************************************************************************2

//	5.	Crop and rearrange.
	
	magnitude(output[0], output[1], mag2);
	//normalize(mag, mag, 1, 0, CV_MINMAX);
	//imshow("after swap", mag);//******************************************************************************************3
	//namedWindow("after IDFT", 0);
	//mag2 = mag;

	//	6.	apply filter 

	//createTrackbar("sigma", "after IDFT", &sigma, 100);
	// make filter first 
	Mat ideal_low_pass(padded.size(), CV_32FC1);
	float d0 = sigma;//The smaller the radius D0, the larger the blur; the larger the radius D0, the smaller the blur
	for (int i = 0; i < padded.rows; i++)
	{
		for (int j = 0; j < padded.cols; j++)
		{
			double d = sqrt(pow((i - padded.rows / 2), 2) + pow((j - padded.cols / 2), 2));//Molecule, Calculate pow must be float type
			if (d <= d0)
			{
				ideal_low_pass.at<float>(i, j) = 1;
			}
			else
			{
				ideal_low_pass.at<float>(i, j) = 0;
			}
		}
	}
	ideal_low_pass_ret = ideal_low_pass;
	//imshow("ideal low pass", ideal_low_pass);//*****************************************************************************************4
	//then apply filter on both real and imaginary parts.
	Mat blur_r, blur_i, complexF;
	multiply(output[0], ideal_low_pass, blur_r); //filter (the real part is multiplied by the corresponding element of the filter template)
	multiply(output[1], ideal_low_pass, blur_i); // filter (imaginary part is multiplied by the corresponding element of the filter template)
	Mat plane1[] = { blur_r, blur_i };
	merge(plane1, 2, complexF);//The real and imaginary parts merge
//	7.	Take inverse
	idft(complexF, complexF); //idft result is also plural
	split(complexF, output);
	magnitude(output[0], output[1], output[0]);

	//	8.normalized for easy display

	normalize(output[0], output_ret, 1, 0, CV_MINMAX);
	//output_ret = output[0];
	//imshow("after IDFT", output[0]);//****************************************************************************************************5
	//waitKey(0);
//return padded;-------------1
//return mag;----------------2
//return mag2;---------------3
//return ideal_low_pass;-----4
//return output[0];----------5
	if (returned_one == 1)
	{
		/*cvtColor(padded1, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		CV_32FC1*/
		/*padded.convertTo(padded, CV_8UC1);
		cvtColor(padded, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255);
		return disp;*/
		return padded;
	}
	else if (returned_one == 2)
	{
		cvtColor(mag, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		//return mag;
	}
	else if (returned_one == 3)
	{
		cvtColor(mag2, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		//return mag2;
	}
	else if (returned_one == 4)
	{
		cvtColor(ideal_low_pass_ret, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		//return ideal_low_pass_ret;
	}
	else if (returned_one == 5)
	{
		cvtColor(output_ret, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp, 255, 0);
		return disp;
		//return output_ret;
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat segmentation(Mat input, int threshold_loc)
{
	
		//Mat input = imread("xxx.jpg", 0);
		Mat im, disp;
		cvtColor(input, im, CV_BGR2GRAY);
		Mat output_guassian(im.rows, im.cols, CV_8UC1);
		Mat output_threshold(im.rows, im.cols, CV_8UC1);
		Mat output_edge(im.rows, im.cols, CV_8UC1);
		//namedWindow("orginal", 0);
		//imshow("orginal", input);---------------------------------------------------------------------------------------------1
		///////////////////////3x3 gaussian filter///////////////////// 

		for (int i = 1; i < im.rows - 1; i++) {
			for (int j = 1; j < im.cols - 1; j++)
			{
				output_guassian.at<uchar>(i, j) = 1.0 / 16.0 *
					(4 * im.at<uchar>(i, j) + 2 * im.at<uchar>(i - 1, j) + 2 * im.at<uchar>(i + 1, j) +
						2 * im.at<uchar>(i, j - 1) + 2 * im.at<uchar>(i, j + 1) + im.at<uchar>(i - 1, j - 1) +
						im.at<uchar>(i - 1, j + 1) + im.at<uchar>(i + 1, j - 1) + im.at<uchar>(i + 1, j + 1));
			}
		}

		////////////Laplace Derivative filter //////////////////////////////  
		for (int i = 1; i < im.rows - 1; i++) {
			for (int j = 1; j < im.cols - 1; j++)
			{
				int sum = 4 * output_guassian.at<uchar>(i, j) - output_guassian.at<uchar>(i - 1, j) - output_guassian.at<uchar>(i + 1, j) - output_guassian.at<uchar>(i, j - 1) - output_guassian.at<uchar>(i, j + 1);
				if (sum < 0)
					sum = 0;
				else if (sum > 255)
					sum = 255;
				output_edge.at<uchar>(i, j) = sum;
			}
		}
		/*namedWindow("smoothed image", 0);
		imshow("smoothed image", output_guassian);----------------------------------------------------------------------------------2
		namedWindow("edge based segmentation", 0);
		imshow("edge based segmentation", output_edge);-----------------------------------------------------------------------------3
		waitKey();*/


		///////////threshold based segmentation////////////////////////// 

		int threshold = threshold_loc;// 154;
		int maxval = 255;
		//namedWindow("threshold based segmentation", 0);

		//createTrackbar("Threshold", "threshold based segmentation", &threshold, 255);---------------------------- track bar

		int x = 0;
		while (x < 10)
		{
			for (int r = 0; r < im.rows; r++)
			{
				for (int c = 0; c < im.cols; c++)
				{
					if (output_guassian.at<uchar>(r, c) > threshold)
						output_threshold.at<uchar>(r, c) = 255;
					else
						output_threshold.at<uchar>(r, c) = 0;
				}

			}
			//imshow("threshold based segmentation", output_threshold);--------------------------------------------------------------4
			//waitKey(0);
			x++;
		}
		cvtColor(output_threshold, disp, CV_GRAY2BGR);
		convertScaleAbs(disp, disp);
		return disp;
}

void sorter(vector<float> probabilities) {
	float swapper;
	for (int i = 1; i < probabilities.size(); i++)
	{
		for (int j = 1; i < probabilities.size(); i++)
		{
			if (probabilities.at(j) > probabilities.at(i))
			{
				swapper = probabilities.at(j);
				probabilities.at(j) = probabilities.at(i);
				probabilities.at(i) = swapper;
			}
		}
	}
}
void huffman_encryption(Mat im, string Save_destination)
{
	Mat input;
	cvtColor(im, input, CV_BGR2GRAY);
	ofstream myfile;
	vector<float> gray_values;
	vector<float> gray_count;
	vector<float> probabilities;
	vector<float> probabilities_copy;
	vector<float> code_summed;
	string output_code;
	float hist[256] = { 0 };
	float gray_value;
	int value_found_flag;
	int initial_size;
	// CODE EL ENCRYPTION

	for (int color = 0; color < 256; color++)
	{
		for (int r = 0; r < input.rows; r++)
		{
			for (int c = 0; c < input.cols; c++)
			{

				if (input.at<uchar>(r, c) == color) {

					hist[color] += 1;

				}
			}
		}
	}
	for (int i = 0; i < 256; i++) {
		if (hist[i] != 0) {
			gray_values.push_back(i);
			gray_count.push_back(hist[i]);
		}
	}
	for (int i = 0; i < gray_values.size(); i++)
	{
		probabilities.push_back(gray_values.at(i) / input.total());
	}
	initial_size = probabilities.size();
	code_summed.push_back(1);
	for (int i = 1; i > probabilities.size(); i++) {
		code_summed.push_back(0);
	}
	probabilities_copy = probabilities;
	for (int i = 0; i < initial_size - 2; i++)
	{
		sorter(probabilities);
		probabilities.at(probabilities.size() - 2) = probabilities.at(probabilities.size() - 1) + probabilities.at(probabilities.size() - 2);

	}

	for (int i = 0; i < initial_size; i++)

	{
		if (i >= 2)
		{
			code_summed.at(i) += 2;
			code_summed.push_back(code_summed.at(i - 1) + 1);

		}
		else {
			code_summed.push_back(code_summed.at(i) + 1);
		}
	}



	//END CODE ENCRYPTION
	myfile.open(Save_destination);
	for (int i = 0; i < code_summed.size(); i++)
	{
		bitset<8> b(code_summed.at(i));
		string added_string = b.to_string();
		stringstream ss;
		ss << added_string;
		int added;
		ss >> added;
		output_code = output_code + "" + added_string + "\n";
	}
	myfile << output_code;
	myfile.close();

}

Mat sobel_edge_detection(Mat img_name)
{
	int pv, ph, sv, sh, s_mag, p_mag;
	Mat input, disp;
	cvtColor(img_name, input, CV_BGR2GRAY);
	//Mat input = imread(img_name, 0);
	Mat sh_output(input.rows, input.cols, CV_8UC1, Scalar(0));
	Mat sv_output(input.rows, input.cols, CV_8UC1, Scalar(0));
	Mat s_output(input.rows, input.cols, CV_8UC1, Scalar(0));
	Mat ph_output(input.rows, input.cols, CV_8UC1, Scalar(0));
	Mat pv_output(input.rows, input.cols, CV_8UC1, Scalar(0));
	Mat p_output(input.rows, input.cols, CV_8UC1, Scalar(0));

	//////////////////////sobel ///////////////////////////////
	for (int i = 1; i < input.rows - 1; i++)
		for (int j = 1; j < input.cols - 1; j++)
		{

			sh_output.at<uchar>(i, j) = abs(input.at<uchar>(i - 1, j - 1) * 1 +
				input.at<uchar>(i - 1, j) * 2 + input.at<uchar>(i - 1, j + 1) * 1 +
				input.at<uchar>(i, j - 1) * 0 + input.at<uchar>(i, j) * 0 +
				input.at<uchar>(i, j + 1) * 0 + input.at<uchar>(i + 1, j - 1) * -1 +
				input.at<uchar>(i + 1, j) * -2 + input.at<uchar>(i + 1, j + 1) * -1);
			sv_output.at<uchar>(i, j) = abs(input.at<uchar>(i - 1, j - 1) * 1 +
				input.at<uchar>(i - 1, j) * 0 + input.at<uchar>(i - 1, j + 1) * -1 +
				input.at<uchar>(i, j - 1) * 2 + input.at<uchar>(i, j) * 0 +
				input.at<uchar>(i, j + 1) * -2 + input.at<uchar>(i + 1, j - 1) * 1 +
				input.at<uchar>(i + 1, j) * 0 + input.at<uchar>(i + 1, j + 1) * -1);
			/////////detect edges in both derection//////////////////////
			sh = sh_output.at<uchar>(i, j);
			sv = sv_output.at<uchar>(i, j);
			s_mag = sqrt((sv * sv) + (sh * sh));
			// s_output.at<uchar>(i, j) = s_mag;
			if (s_mag > 100)
				s_output.at<uchar>(i, j) = 255;
			else
				s_output.at<uchar>(i, j) = 0;
		}

	/*namedWindow("sobel_edges detection", 0);
	imshow("sobel_edges detection", s_output);
	waitKey(0);*/
	cvtColor(s_output, disp, CV_GRAY2BGR);
	convertScaleAbs(disp, disp);
	return disp;
}

	/*private: cv::Mat HistBalance(Mat inputMat) {
		Mat hsv, disp;
		cvtColor(inputMat, hsv, CV_BGR2HSV);  // Convert BGR to HSV
		vector<Mat> hsv_channels;
		split(hsv, hsv_channels);				// Get the V channel
		equalizeHist(hsv_channels[2], hsv_channels[2]);		// Balance V Channel
		merge(hsv_channels, hsv); // merge V channel into hsv image
		cvtColor(hsv, disp, CV_HSV2BGR); // Covert back to BGR image
		convertScaleAbs(disp, disp);
		return disp;
	 }*/

	/// <summary>
	/// Summary for MainForm
	/// </summary>
	public ref class MainForm : public System::Windows::Forms::Form
	{

	
	public:
		MainForm(void)
		{
			
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MainForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::TabControl^  btnBrowse3;
	protected:
	private: System::Windows::Forms::TabPage^ tabPage2;
	private: System::Windows::Forms::Label^ label11;
	private: System::Windows::Forms::Label^ label10;
	private: System::Windows::Forms::Button^ button7;
	private: System::Windows::Forms::PictureBox^ pct_tar_op;
	private: System::Windows::Forms::PictureBox^ pct_scr_op;


	private: System::Windows::Forms::Button^ btn_zoom_op;

	private: System::Windows::Forms::Button^ btn_rot_270_op;

	private: System::Windows::Forms::Button^ btn_rot_180_op;

	private: System::Windows::Forms::Button^ btn_rot_90_op;

	private: System::Windows::Forms::Button^ btn_flp_op;

	private: System::Windows::Forms::Label^ label12;
	private: System::Windows::Forms::TabPage^ tabPage4;
	private: System::Windows::Forms::FlowLayoutPanel^ flowLayoutPanel1;
	private: System::Windows::Forms::Label^ label14;
	private: System::Windows::Forms::Button^ btn_avg_filter;

	private: System::Windows::Forms::Label^ label13;
	private: System::Windows::Forms::PictureBox^ pct_tar_filter;
	private: System::Windows::Forms::PictureBox^ pct_scr_filter;
	private: System::Windows::Forms::Button^ btn_lod_filters;
	private: System::Windows::Forms::Label^ label15;
	private: System::Windows::Forms::Label^ label16;
	private: System::Windows::Forms::Button^ btn_mid_filter;
	private: System::Windows::Forms::Splitter^ splitter1;
	private: System::Windows::Forms::Label^ label17;
	private: System::Windows::Forms::Button^ btc_pre_filter;
	private: System::Windows::Forms::Button^ btc_sob_filter;
	private: System::Windows::Forms::TabPage^ tabPage5;
	private: System::Windows::Forms::FlowLayoutPanel^ flowLayoutPanel2;
	private: System::Windows::Forms::Button^ btn_prt_adj;
	private: System::Windows::Forms::Button^ btn_blin;
	private: System::Windows::Forms::Button^ btn_hist;
	private: System::Windows::Forms::Button^ btn_log_trn;
	private: System::Windows::Forms::Button^ btn_pwr_low;
	private: System::Windows::Forms::Button^ btn_neg;
	private: System::Windows::Forms::TrackBar^ brt_adj_trackBar;
	private: System::Windows::Forms::Label^ label18;
	private: System::Windows::Forms::Label^ label21;
	private: System::Windows::Forms::Label^ label20;
	private: System::Windows::Forms::Label^ label19;
	private: System::Windows::Forms::PictureBox^ pct_trg_pp;

private: System::Windows::Forms::PictureBox^ pct_scr_pp;

	private: System::Windows::Forms::Button^ btn_lod_pp;
private: System::Windows::Forms::Button^ btn_lod_pp2;
private: System::Windows::Forms::PictureBox^ pct_scr_pp2;
private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::FlowLayoutPanel^ flowLayoutPanel3;
	private: System::Windows::Forms::TrackBar^ rot_adj_trackBar;
	private: System::Windows::Forms::Button^ btn_rot_rnm_op;
	private: System::Windows::Forms::CheckBox^ overwrite_box;
	private: System::Windows::Forms::CheckBox^ over_write_filter;
	private: System::Windows::Forms::TabPage^ freq_tab;
	private: System::Windows::Forms::Label^ label7;
	private: System::Windows::Forms::Label^ label6;
	private: System::Windows::Forms::Label^ label5;
	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::PictureBox^ pic_flt_ilp4;
	private: System::Windows::Forms::PictureBox^ pic_flt_aswp3;
	private: System::Windows::Forms::PictureBox^ pic_flt_bswp2;
	private: System::Windows::Forms::PictureBox^ pic_flt_IDFT5;
	private: System::Windows::Forms::PictureBox^ pic_flt_pad1;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::TrackBar^ frq_smo_flt_trackBar;
	private: System::Windows::Forms::FlowLayoutPanel^ flowLayoutPanel4;
	private: System::Windows::Forms::Button^ btn_frq_smo_flt;
	private: System::Windows::Forms::Button^ btn_frq_shr_flt;
	private: System::Windows::Forms::Label^ label1;
private: System::Windows::Forms::Button^ btn_lod_fd;
private: System::Windows::Forms::Label^ label8;
private: System::Windows::Forms::PictureBox^ pct_scr_fd;
private: System::Windows::Forms::CheckBox^ over_write_freq;
private: System::Windows::Forms::Label^ label9;
private: System::Windows::Forms::TextBox^ textBox2;
private: System::Windows::Forms::TabPage^ tabPage1;
private: System::Windows::Forms::TabPage^ tabPage3;
private: System::Windows::Forms::FlowLayoutPanel^ flowLayoutPanel5;
private: System::Windows::Forms::Button^ btn_seg;

private: System::Windows::Forms::Button^ btn_lod_seg;
private: System::Windows::Forms::Label^ label22;
private: System::Windows::Forms::Label^ label24;
private: System::Windows::Forms::Label^ label23;
private: System::Windows::Forms::PictureBox^ pct_trg_seg;
private: System::Windows::Forms::PictureBox^ pct_scr_seg;
private: System::Windows::Forms::Label^ label25;
private: System::Windows::Forms::TrackBar^ thr_adj_trackBar;
private: System::Windows::Forms::CheckBox^ over_write_seg;
private: System::Windows::Forms::Label^ label26;
private: System::Windows::Forms::TextBox^ textBox3;
private: System::Windows::Forms::Label^ label28;
private: System::Windows::Forms::TextBox^ textBox4;
private: System::Windows::Forms::Label^ label27;
private: System::Windows::Forms::Label^ label29;
private: System::Windows::Forms::TextBox^ textBox5;
private: System::Windows::Forms::CheckBox^ checkBox7;

private: System::Windows::Forms::Button^ button1;





	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

	

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(MainForm::typeid));
			this->btnBrowse3 = (gcnew System::Windows::Forms::TabControl());
			this->tabPage2 = (gcnew System::Windows::Forms::TabPage());
			this->label29 = (gcnew System::Windows::Forms::Label());
			this->textBox5 = (gcnew System::Windows::Forms::TextBox());
			this->label28 = (gcnew System::Windows::Forms::Label());
			this->textBox4 = (gcnew System::Windows::Forms::TextBox());
			this->label27 = (gcnew System::Windows::Forms::Label());
			this->overwrite_box = (gcnew System::Windows::Forms::CheckBox());
			this->rot_adj_trackBar = (gcnew System::Windows::Forms::TrackBar());
			this->flowLayoutPanel3 = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->btn_flp_op = (gcnew System::Windows::Forms::Button());
			this->btn_rot_90_op = (gcnew System::Windows::Forms::Button());
			this->btn_rot_180_op = (gcnew System::Windows::Forms::Button());
			this->btn_rot_270_op = (gcnew System::Windows::Forms::Button());
			this->btn_zoom_op = (gcnew System::Windows::Forms::Button());
			this->btn_rot_rnm_op = (gcnew System::Windows::Forms::Button());
			this->label12 = (gcnew System::Windows::Forms::Label());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->button7 = (gcnew System::Windows::Forms::Button());
			this->pct_tar_op = (gcnew System::Windows::Forms::PictureBox());
			this->pct_scr_op = (gcnew System::Windows::Forms::PictureBox());
			this->tabPage4 = (gcnew System::Windows::Forms::TabPage());
			this->label16 = (gcnew System::Windows::Forms::Label());
			this->label15 = (gcnew System::Windows::Forms::Label());
			this->btn_lod_filters = (gcnew System::Windows::Forms::Button());
			this->pct_tar_filter = (gcnew System::Windows::Forms::PictureBox());
			this->pct_scr_filter = (gcnew System::Windows::Forms::PictureBox());
			this->flowLayoutPanel1 = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->label14 = (gcnew System::Windows::Forms::Label());
			this->btn_avg_filter = (gcnew System::Windows::Forms::Button());
			this->btn_mid_filter = (gcnew System::Windows::Forms::Button());
			this->splitter1 = (gcnew System::Windows::Forms::Splitter());
			this->label17 = (gcnew System::Windows::Forms::Label());
			this->btc_pre_filter = (gcnew System::Windows::Forms::Button());
			this->btc_sob_filter = (gcnew System::Windows::Forms::Button());
			this->label13 = (gcnew System::Windows::Forms::Label());
			this->over_write_filter = (gcnew System::Windows::Forms::CheckBox());
			this->tabPage5 = (gcnew System::Windows::Forms::TabPage());
			this->checkBox7 = (gcnew System::Windows::Forms::CheckBox());
			this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			this->btn_lod_pp2 = (gcnew System::Windows::Forms::Button());
			this->pct_scr_pp2 = (gcnew System::Windows::Forms::PictureBox());
			this->label21 = (gcnew System::Windows::Forms::Label());
			this->label20 = (gcnew System::Windows::Forms::Label());
			this->label19 = (gcnew System::Windows::Forms::Label());
			this->pct_trg_pp = (gcnew System::Windows::Forms::PictureBox());
			this->pct_scr_pp = (gcnew System::Windows::Forms::PictureBox());
			this->btn_lod_pp = (gcnew System::Windows::Forms::Button());
			this->flowLayoutPanel2 = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->btn_prt_adj = (gcnew System::Windows::Forms::Button());
			this->btn_blin = (gcnew System::Windows::Forms::Button());
			this->btn_hist = (gcnew System::Windows::Forms::Button());
			this->btn_log_trn = (gcnew System::Windows::Forms::Button());
			this->btn_pwr_low = (gcnew System::Windows::Forms::Button());
			this->btn_neg = (gcnew System::Windows::Forms::Button());
			this->brt_adj_trackBar = (gcnew System::Windows::Forms::TrackBar());
			this->label18 = (gcnew System::Windows::Forms::Label());
			this->freq_tab = (gcnew System::Windows::Forms::TabPage());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->textBox2 = (gcnew System::Windows::Forms::TextBox());
			this->over_write_freq = (gcnew System::Windows::Forms::CheckBox());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->pct_scr_fd = (gcnew System::Windows::Forms::PictureBox());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->pic_flt_ilp4 = (gcnew System::Windows::Forms::PictureBox());
			this->pic_flt_aswp3 = (gcnew System::Windows::Forms::PictureBox());
			this->pic_flt_bswp2 = (gcnew System::Windows::Forms::PictureBox());
			this->pic_flt_IDFT5 = (gcnew System::Windows::Forms::PictureBox());
			this->pic_flt_pad1 = (gcnew System::Windows::Forms::PictureBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->frq_smo_flt_trackBar = (gcnew System::Windows::Forms::TrackBar());
			this->flowLayoutPanel4 = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->btn_frq_smo_flt = (gcnew System::Windows::Forms::Button());
			this->btn_frq_shr_flt = (gcnew System::Windows::Forms::Button());
			this->btn_lod_fd = (gcnew System::Windows::Forms::Button());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->tabPage1 = (gcnew System::Windows::Forms::TabPage());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->tabPage3 = (gcnew System::Windows::Forms::TabPage());
			this->label24 = (gcnew System::Windows::Forms::Label());
			this->label23 = (gcnew System::Windows::Forms::Label());
			this->pct_trg_seg = (gcnew System::Windows::Forms::PictureBox());
			this->pct_scr_seg = (gcnew System::Windows::Forms::PictureBox());
			this->over_write_seg = (gcnew System::Windows::Forms::CheckBox());
			this->flowLayoutPanel5 = (gcnew System::Windows::Forms::FlowLayoutPanel());
			this->btn_seg = (gcnew System::Windows::Forms::Button());
			this->btn_lod_seg = (gcnew System::Windows::Forms::Button());
			this->label25 = (gcnew System::Windows::Forms::Label());
			this->thr_adj_trackBar = (gcnew System::Windows::Forms::TrackBar());
			this->label26 = (gcnew System::Windows::Forms::Label());
			this->textBox3 = (gcnew System::Windows::Forms::TextBox());
			this->label22 = (gcnew System::Windows::Forms::Label());
			this->btnBrowse3->SuspendLayout();
			this->tabPage2->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->rot_adj_trackBar))->BeginInit();
			this->flowLayoutPanel3->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_tar_op))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_op))->BeginInit();
			this->tabPage4->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_tar_filter))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_filter))->BeginInit();
			this->flowLayoutPanel1->SuspendLayout();
			this->tabPage5->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_pp2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_trg_pp))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_pp))->BeginInit();
			this->flowLayoutPanel2->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->brt_adj_trackBar))->BeginInit();
			this->freq_tab->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_fd))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_ilp4))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_aswp3))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_bswp2))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_IDFT5))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_pad1))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->frq_smo_flt_trackBar))->BeginInit();
			this->flowLayoutPanel4->SuspendLayout();
			this->tabPage1->SuspendLayout();
			this->tabPage3->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_trg_seg))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_seg))->BeginInit();
			this->flowLayoutPanel5->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->thr_adj_trackBar))->BeginInit();
			this->SuspendLayout();
			// 
			// btnBrowse3
			// 
			this->btnBrowse3->Controls->Add(this->tabPage2);
			this->btnBrowse3->Controls->Add(this->tabPage4);
			this->btnBrowse3->Controls->Add(this->tabPage5);
			this->btnBrowse3->Controls->Add(this->freq_tab);
			this->btnBrowse3->Controls->Add(this->tabPage1);
			this->btnBrowse3->Controls->Add(this->tabPage3);
			this->btnBrowse3->Location = System::Drawing::Point(3, 4);
			this->btnBrowse3->Margin = System::Windows::Forms::Padding(4);
			this->btnBrowse3->Name = L"btnBrowse3";
			this->btnBrowse3->SelectedIndex = 0;
			this->btnBrowse3->Size = System::Drawing::Size(1222, 826);
			this->btnBrowse3->TabIndex = 0;
			this->btnBrowse3->SelectedIndexChanged += gcnew System::EventHandler(this, &MainForm::btnBrowse3_SelectedIndexChanged);
			// 
			// tabPage2
			// 
			this->tabPage2->BackColor = System::Drawing::Color::CornflowerBlue;
			this->tabPage2->Controls->Add(this->label29);
			this->tabPage2->Controls->Add(this->textBox5);
			this->tabPage2->Controls->Add(this->label28);
			this->tabPage2->Controls->Add(this->textBox4);
			this->tabPage2->Controls->Add(this->label27);
			this->tabPage2->Controls->Add(this->overwrite_box);
			this->tabPage2->Controls->Add(this->rot_adj_trackBar);
			this->tabPage2->Controls->Add(this->flowLayoutPanel3);
			this->tabPage2->Controls->Add(this->label12);
			this->tabPage2->Controls->Add(this->label11);
			this->tabPage2->Controls->Add(this->label10);
			this->tabPage2->Controls->Add(this->button7);
			this->tabPage2->Controls->Add(this->pct_tar_op);
			this->tabPage2->Controls->Add(this->pct_scr_op);
			this->tabPage2->Location = System::Drawing::Point(4, 25);
			this->tabPage2->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->tabPage2->Name = L"tabPage2";
			this->tabPage2->Padding = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->tabPage2->Size = System::Drawing::Size(1214, 797);
			this->tabPage2->TabIndex = 3;
			this->tabPage2->Text = L"Image Operations";
			// 
			// label29
			// 
			this->label29->AutoSize = true;
			this->label29->Location = System::Drawing::Point(3, 634);
			this->label29->Name = L"label29";
			this->label29->Size = System::Drawing::Size(39, 17);
			this->label29->TabIndex = 18;
			this->label29->Text = L"Scale";
			// 
			// textBox5
			// 
			this->textBox5->Location = System::Drawing::Point(48, 629);
			this->textBox5->Name = L"textBox5";
			this->textBox5->Size = System::Drawing::Size(88, 24);
			this->textBox5->TabIndex = 17;
			this->textBox5->TextChanged += gcnew System::EventHandler(this, &MainForm::textBox5_TextChanged);
			// 
			// label28
			// 
			this->label28->AutoSize = true;
			this->label28->Location = System::Drawing::Point(4, 606);
			this->label28->Name = L"label28";
			this->label28->Size = System::Drawing::Size(40, 17);
			this->label28->TabIndex = 16;
			this->label28->Text = L"Value";
			// 
			// textBox4
			// 
			this->textBox4->Location = System::Drawing::Point(48, 601);
			this->textBox4->Name = L"textBox4";
			this->textBox4->Size = System::Drawing::Size(88, 24);
			this->textBox4->TabIndex = 15;
			this->textBox4->TextChanged += gcnew System::EventHandler(this, &MainForm::textBox4_TextChanged);
			// 
			// label27
			// 
			this->label27->AutoSize = true;
			this->label27->Location = System::Drawing::Point(9, 516);
			this->label27->Name = L"label27";
			this->label27->Size = System::Drawing::Size(41, 17);
			this->label27->TabIndex = 14;
			this->label27->Text = L"Angel";
			// 
			// overwrite_box
			// 
			this->overwrite_box->AutoSize = true;
			this->overwrite_box->Location = System::Drawing::Point(5, 316);
			this->overwrite_box->Name = L"overwrite_box";
			this->overwrite_box->Size = System::Drawing::Size(108, 21);
			this->overwrite_box->TabIndex = 13;
			this->overwrite_box->Text = L"Over Write \?";
			this->overwrite_box->UseVisualStyleBackColor = true;
			this->overwrite_box->CheckedChanged += gcnew System::EventHandler(this, &MainForm::overwrite_box_CheckedChanged);
			// 
			// rot_adj_trackBar
			// 
			this->rot_adj_trackBar->BackColor = System::Drawing::Color::SteelBlue;
			this->rot_adj_trackBar->Location = System::Drawing::Point(7, 539);
			this->rot_adj_trackBar->Maximum = 360;
			this->rot_adj_trackBar->Name = L"rot_adj_trackBar";
			this->rot_adj_trackBar->Size = System::Drawing::Size(624, 56);
			this->rot_adj_trackBar->TabIndex = 12;
			this->rot_adj_trackBar->Scroll += gcnew System::EventHandler(this, &MainForm::rot_adj_trackBar_Scroll);
			// 
			// flowLayoutPanel3
			// 
			this->flowLayoutPanel3->BackColor = System::Drawing::Color::SteelBlue;
			this->flowLayoutPanel3->Controls->Add(this->btn_flp_op);
			this->flowLayoutPanel3->Controls->Add(this->btn_rot_90_op);
			this->flowLayoutPanel3->Controls->Add(this->btn_rot_180_op);
			this->flowLayoutPanel3->Controls->Add(this->btn_rot_270_op);
			this->flowLayoutPanel3->Controls->Add(this->btn_zoom_op);
			this->flowLayoutPanel3->Controls->Add(this->btn_rot_rnm_op);
			this->flowLayoutPanel3->Location = System::Drawing::Point(4, 23);
			this->flowLayoutPanel3->Name = L"flowLayoutPanel3";
			this->flowLayoutPanel3->Size = System::Drawing::Size(181, 295);
			this->flowLayoutPanel3->TabIndex = 11;
			// 
			// btn_flp_op
			// 
			this->btn_flp_op->Location = System::Drawing::Point(3, 2);
			this->btn_flp_op->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btn_flp_op->Name = L"btn_flp_op";
			this->btn_flp_op->Size = System::Drawing::Size(172, 39);
			this->btn_flp_op->TabIndex = 0;
			this->btn_flp_op->Text = L"Flip";
			this->btn_flp_op->UseVisualStyleBackColor = true;
			this->btn_flp_op->Click += gcnew System::EventHandler(this, &MainForm::button2_Click);
			// 
			// btn_rot_90_op
			// 
			this->btn_rot_90_op->Location = System::Drawing::Point(3, 45);
			this->btn_rot_90_op->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btn_rot_90_op->Name = L"btn_rot_90_op";
			this->btn_rot_90_op->Size = System::Drawing::Size(172, 39);
			this->btn_rot_90_op->TabIndex = 1;
			this->btn_rot_90_op->Text = L"Rotation 90";
			this->btn_rot_90_op->UseVisualStyleBackColor = true;
			this->btn_rot_90_op->Click += gcnew System::EventHandler(this, &MainForm::button3_Click);
			// 
			// btn_rot_180_op
			// 
			this->btn_rot_180_op->Location = System::Drawing::Point(3, 88);
			this->btn_rot_180_op->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btn_rot_180_op->Name = L"btn_rot_180_op";
			this->btn_rot_180_op->Size = System::Drawing::Size(172, 39);
			this->btn_rot_180_op->TabIndex = 2;
			this->btn_rot_180_op->Text = L"Rotation 180";
			this->btn_rot_180_op->UseVisualStyleBackColor = true;
			this->btn_rot_180_op->Click += gcnew System::EventHandler(this, &MainForm::button4_Click);
			// 
			// btn_rot_270_op
			// 
			this->btn_rot_270_op->Location = System::Drawing::Point(3, 131);
			this->btn_rot_270_op->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btn_rot_270_op->Name = L"btn_rot_270_op";
			this->btn_rot_270_op->Size = System::Drawing::Size(172, 39);
			this->btn_rot_270_op->TabIndex = 3;
			this->btn_rot_270_op->Text = L"Rotation 270";
			this->btn_rot_270_op->UseVisualStyleBackColor = true;
			this->btn_rot_270_op->Click += gcnew System::EventHandler(this, &MainForm::btn_rot_270_op_Click);
			// 
			// btn_zoom_op
			// 
			this->btn_zoom_op->Location = System::Drawing::Point(3, 174);
			this->btn_zoom_op->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btn_zoom_op->Name = L"btn_zoom_op";
			this->btn_zoom_op->Size = System::Drawing::Size(172, 39);
			this->btn_zoom_op->TabIndex = 4;
			this->btn_zoom_op->Text = L"Zoom";
			this->btn_zoom_op->UseVisualStyleBackColor = true;
			this->btn_zoom_op->Click += gcnew System::EventHandler(this, &MainForm::btn_zoom_op_Click);
			// 
			// btn_rot_rnm_op
			// 
			this->btn_rot_rnm_op->Location = System::Drawing::Point(3, 218);
			this->btn_rot_rnm_op->Name = L"btn_rot_rnm_op";
			this->btn_rot_rnm_op->Size = System::Drawing::Size(172, 39);
			this->btn_rot_rnm_op->TabIndex = 5;
			this->btn_rot_rnm_op->Text = L"Rotation With Any Value";
			this->btn_rot_rnm_op->UseVisualStyleBackColor = true;
			this->btn_rot_rnm_op->Click += gcnew System::EventHandler(this, &MainForm::btn_rot_rnm_op_Click);
			// 
			// label12
			// 
			this->label12->AutoSize = true;
			this->label12->Location = System::Drawing::Point(6, 2);
			this->label12->Name = L"label12";
			this->label12->Size = System::Drawing::Size(74, 17);
			this->label12->TabIndex = 10;
			this->label12->Text = L"Operations";
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->Location = System::Drawing::Point(557, 23);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(93, 17);
			this->label11->TabIndex = 9;
			this->label11->Text = L"Source Image";
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->Location = System::Drawing::Point(560, 438);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(90, 17);
			this->label10->TabIndex = 8;
			this->label10->Text = L"Target Image";
			// 
			// button7
			// 
			this->button7->Location = System::Drawing::Point(4, 342);
			this->button7->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->button7->Name = L"button7";
			this->button7->Size = System::Drawing::Size(172, 39);
			this->button7->TabIndex = 7;
			this->button7->Text = L"Load Image";
			this->button7->UseVisualStyleBackColor = true;
			this->button7->Click += gcnew System::EventHandler(this, &MainForm::button7_Click);
			// 
			// pct_tar_op
			// 
			this->pct_tar_op->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_tar_op->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_tar_op->Location = System::Drawing::Point(646, 438);
			this->pct_tar_op->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->pct_tar_op->Name = L"pct_tar_op";
			this->pct_tar_op->Size = System::Drawing::Size(563, 349);
			this->pct_tar_op->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_tar_op->TabIndex = 6;
			this->pct_tar_op->TabStop = false;
			// 
			// pct_scr_op
			// 
			this->pct_scr_op->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_scr_op->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_scr_op->Location = System::Drawing::Point(646, 23);
			this->pct_scr_op->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->pct_scr_op->Name = L"pct_scr_op";
			this->pct_scr_op->Size = System::Drawing::Size(563, 349);
			this->pct_scr_op->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_scr_op->TabIndex = 5;
			this->pct_scr_op->TabStop = false;
			// 
			// tabPage4
			// 
			this->tabPage4->BackColor = System::Drawing::Color::CornflowerBlue;
			this->tabPage4->Controls->Add(this->label16);
			this->tabPage4->Controls->Add(this->label15);
			this->tabPage4->Controls->Add(this->btn_lod_filters);
			this->tabPage4->Controls->Add(this->pct_tar_filter);
			this->tabPage4->Controls->Add(this->pct_scr_filter);
			this->tabPage4->Controls->Add(this->flowLayoutPanel1);
			this->tabPage4->Controls->Add(this->label13);
			this->tabPage4->Controls->Add(this->over_write_filter);
			this->tabPage4->Location = System::Drawing::Point(4, 25);
			this->tabPage4->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->tabPage4->Name = L"tabPage4";
			this->tabPage4->Padding = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->tabPage4->Size = System::Drawing::Size(1214, 797);
			this->tabPage4->TabIndex = 4;
			this->tabPage4->Text = L"Filters";
			// 
			// label16
			// 
			this->label16->AutoSize = true;
			this->label16->Location = System::Drawing::Point(428, 415);
			this->label16->Name = L"label16";
			this->label16->Size = System::Drawing::Size(90, 17);
			this->label16->TabIndex = 6;
			this->label16->Text = L"Target Image";
			this->label16->Click += gcnew System::EventHandler(this, &MainForm::label16_Click);
			// 
			// label15
			// 
			this->label15->AutoSize = true;
			this->label15->Location = System::Drawing::Point(428, 12);
			this->label15->Name = L"label15";
			this->label15->Size = System::Drawing::Size(93, 17);
			this->label15->TabIndex = 5;
			this->label15->Text = L"Source Image";
			// 
			// btn_lod_filters
			// 
			this->btn_lod_filters->Cursor = System::Windows::Forms::Cursors::AppStarting;
			this->btn_lod_filters->Location = System::Drawing::Point(10, 327);
			this->btn_lod_filters->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btn_lod_filters->Name = L"btn_lod_filters";
			this->btn_lod_filters->Size = System::Drawing::Size(172, 39);
			this->btn_lod_filters->TabIndex = 4;
			this->btn_lod_filters->Text = L"Load Image";
			this->btn_lod_filters->UseVisualStyleBackColor = true;
			this->btn_lod_filters->Click += gcnew System::EventHandler(this, &MainForm::btn_lod_filters_Click);
			// 
			// pct_tar_filter
			// 
			this->pct_tar_filter->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_tar_filter->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_tar_filter->Cursor = System::Windows::Forms::Cursors::Cross;
			this->pct_tar_filter->Location = System::Drawing::Point(430, 433);
			this->pct_tar_filter->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->pct_tar_filter->Name = L"pct_tar_filter";
			this->pct_tar_filter->Size = System::Drawing::Size(563, 349);
			this->pct_tar_filter->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_tar_filter->TabIndex = 3;
			this->pct_tar_filter->TabStop = false;
			// 
			// pct_scr_filter
			// 
			this->pct_scr_filter->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_scr_filter->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_scr_filter->Cursor = System::Windows::Forms::Cursors::Cross;
			this->pct_scr_filter->Location = System::Drawing::Point(430, 32);
			this->pct_scr_filter->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->pct_scr_filter->Name = L"pct_scr_filter";
			this->pct_scr_filter->Size = System::Drawing::Size(563, 349);
			this->pct_scr_filter->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_scr_filter->TabIndex = 2;
			this->pct_scr_filter->TabStop = false;
			// 
			// flowLayoutPanel1
			// 
			this->flowLayoutPanel1->BackColor = System::Drawing::Color::SteelBlue;
			this->flowLayoutPanel1->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->flowLayoutPanel1->Controls->Add(this->label14);
			this->flowLayoutPanel1->Controls->Add(this->btn_avg_filter);
			this->flowLayoutPanel1->Controls->Add(this->btn_mid_filter);
			this->flowLayoutPanel1->Controls->Add(this->splitter1);
			this->flowLayoutPanel1->Controls->Add(this->label17);
			this->flowLayoutPanel1->Controls->Add(this->btc_pre_filter);
			this->flowLayoutPanel1->Controls->Add(this->btc_sob_filter);
			this->flowLayoutPanel1->Cursor = System::Windows::Forms::Cursors::Hand;
			this->flowLayoutPanel1->Location = System::Drawing::Point(6, 32);
			this->flowLayoutPanel1->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->flowLayoutPanel1->Name = L"flowLayoutPanel1";
			this->flowLayoutPanel1->Size = System::Drawing::Size(181, 256);
			this->flowLayoutPanel1->TabIndex = 1;
			// 
			// label14
			// 
			this->label14->AutoSize = true;
			this->label14->Location = System::Drawing::Point(3, 0);
			this->label14->Name = L"label14";
			this->label14->Size = System::Drawing::Size(154, 17);
			this->label14->TabIndex = 2;
			this->label14->Text = L"Smoothing spatial Filters";
			// 
			// btn_avg_filter
			// 
			this->btn_avg_filter->Cursor = System::Windows::Forms::Cursors::AppStarting;
			this->btn_avg_filter->Location = System::Drawing::Point(3, 19);
			this->btn_avg_filter->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btn_avg_filter->Name = L"btn_avg_filter";
			this->btn_avg_filter->Size = System::Drawing::Size(172, 39);
			this->btn_avg_filter->TabIndex = 2;
			this->btn_avg_filter->Text = L"Averaging Filter";
			this->btn_avg_filter->UseVisualStyleBackColor = true;
			this->btn_avg_filter->Click += gcnew System::EventHandler(this, &MainForm::btn_avg_filter_Click);
			// 
			// btn_mid_filter
			// 
			this->btn_mid_filter->Cursor = System::Windows::Forms::Cursors::AppStarting;
			this->btn_mid_filter->Location = System::Drawing::Point(3, 62);
			this->btn_mid_filter->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btn_mid_filter->Name = L"btn_mid_filter";
			this->btn_mid_filter->Size = System::Drawing::Size(172, 39);
			this->btn_mid_filter->TabIndex = 7;
			this->btn_mid_filter->Text = L"Median Filter";
			this->btn_mid_filter->UseVisualStyleBackColor = true;
			this->btn_mid_filter->Click += gcnew System::EventHandler(this, &MainForm::btn_mid_filter_Click);
			// 
			// splitter1
			// 
			this->splitter1->BackColor = System::Drawing::SystemColors::ButtonShadow;
			this->splitter1->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->splitter1->Location = System::Drawing::Point(3, 105);
			this->splitter1->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->splitter1->Name = L"splitter1";
			this->splitter1->Size = System::Drawing::Size(174, 0);
			this->splitter1->TabIndex = 8;
			this->splitter1->TabStop = false;
			// 
			// label17
			// 
			this->label17->AutoSize = true;
			this->label17->Location = System::Drawing::Point(3, 107);
			this->label17->Name = L"label17";
			this->label17->Size = System::Drawing::Size(158, 17);
			this->label17->TabIndex = 9;
			this->label17->Text = L"Sharpening Spatial Filters";
			// 
			// btc_pre_filter
			// 
			this->btc_pre_filter->Cursor = System::Windows::Forms::Cursors::AppStarting;
			this->btc_pre_filter->Location = System::Drawing::Point(3, 126);
			this->btc_pre_filter->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btc_pre_filter->Name = L"btc_pre_filter";
			this->btc_pre_filter->Size = System::Drawing::Size(172, 39);
			this->btc_pre_filter->TabIndex = 10;
			this->btc_pre_filter->Text = L"Prewitt differential Filter";
			this->btc_pre_filter->UseVisualStyleBackColor = true;
			this->btc_pre_filter->Click += gcnew System::EventHandler(this, &MainForm::btc_pre_filter_Click);
			// 
			// btc_sob_filter
			// 
			this->btc_sob_filter->Location = System::Drawing::Point(3, 169);
			this->btc_sob_filter->Margin = System::Windows::Forms::Padding(3, 2, 3, 2);
			this->btc_sob_filter->Name = L"btc_sob_filter";
			this->btc_sob_filter->Size = System::Drawing::Size(172, 39);
			this->btc_sob_filter->TabIndex = 11;
			this->btc_sob_filter->Text = L"Sobel Filter";
			this->btc_sob_filter->UseVisualStyleBackColor = true;
			this->btc_sob_filter->Click += gcnew System::EventHandler(this, &MainForm::btc_sob_filter_Click);
			// 
			// label13
			// 
			this->label13->AutoSize = true;
			this->label13->Location = System::Drawing::Point(25, 12);
			this->label13->Name = L"label13";
			this->label13->Size = System::Drawing::Size(79, 17);
			this->label13->TabIndex = 0;
			this->label13->Text = L"Filters Menu";
			// 
			// over_write_filter
			// 
			this->over_write_filter->AutoSize = true;
			this->over_write_filter->Location = System::Drawing::Point(10, 293);
			this->over_write_filter->Name = L"over_write_filter";
			this->over_write_filter->Size = System::Drawing::Size(104, 21);
			this->over_write_filter->TabIndex = 7;
			this->over_write_filter->Text = L"Over write \?";
			this->over_write_filter->UseVisualStyleBackColor = true;
			this->over_write_filter->CheckedChanged += gcnew System::EventHandler(this, &MainForm::over_write_filter_CheckedChanged);
			// 
			// tabPage5
			// 
			this->tabPage5->BackColor = System::Drawing::Color::CornflowerBlue;
			this->tabPage5->Controls->Add(this->checkBox7);
			this->tabPage5->Controls->Add(this->textBox1);
			this->tabPage5->Controls->Add(this->btn_lod_pp2);
			this->tabPage5->Controls->Add(this->pct_scr_pp2);
			this->tabPage5->Controls->Add(this->label21);
			this->tabPage5->Controls->Add(this->label20);
			this->tabPage5->Controls->Add(this->label19);
			this->tabPage5->Controls->Add(this->pct_trg_pp);
			this->tabPage5->Controls->Add(this->pct_scr_pp);
			this->tabPage5->Controls->Add(this->btn_lod_pp);
			this->tabPage5->Controls->Add(this->flowLayoutPanel2);
			this->tabPage5->Controls->Add(this->brt_adj_trackBar);
			this->tabPage5->Controls->Add(this->label18);
			this->tabPage5->Location = System::Drawing::Point(4, 25);
			this->tabPage5->Margin = System::Windows::Forms::Padding(4);
			this->tabPage5->Name = L"tabPage5";
			this->tabPage5->Padding = System::Windows::Forms::Padding(4);
			this->tabPage5->Size = System::Drawing::Size(1214, 797);
			this->tabPage5->TabIndex = 5;
			this->tabPage5->Text = L"Point Processing";
			// 
			// checkBox7
			// 
			this->checkBox7->AutoSize = true;
			this->checkBox7->Location = System::Drawing::Point(310, 71);
			this->checkBox7->Name = L"checkBox7";
			this->checkBox7->Size = System::Drawing::Size(131, 21);
			this->checkBox7->TabIndex = 11;
			this->checkBox7->Text = L"over write filter \?";
			this->checkBox7->UseVisualStyleBackColor = true;
			this->checkBox7->CheckedChanged += gcnew System::EventHandler(this, &MainForm::checkBox1_CheckedChanged);
			// 
			// textBox1
			// 
			this->textBox1->Location = System::Drawing::Point(264, 409);
			this->textBox1->Name = L"textBox1";
			this->textBox1->Size = System::Drawing::Size(88, 24);
			this->textBox1->TabIndex = 10;
			// 
			// btn_lod_pp2
			// 
			this->btn_lod_pp2->Location = System::Drawing::Point(11, 740);
			this->btn_lod_pp2->Margin = System::Windows::Forms::Padding(4);
			this->btn_lod_pp2->Name = L"btn_lod_pp2";
			this->btn_lod_pp2->Size = System::Drawing::Size(220, 42);
			this->btn_lod_pp2->TabIndex = 9;
			this->btn_lod_pp2->Text = L"Load Second Image";
			this->btn_lod_pp2->UseVisualStyleBackColor = true;
			this->btn_lod_pp2->Click += gcnew System::EventHandler(this, &MainForm::btn_lod_pp2_Click);
			// 
			// pct_scr_pp2
			// 
			this->pct_scr_pp2->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_scr_pp2->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_scr_pp2->Location = System::Drawing::Point(8, 526);
			this->pct_scr_pp2->Margin = System::Windows::Forms::Padding(4);
			this->pct_scr_pp2->Name = L"pct_scr_pp2";
			this->pct_scr_pp2->Size = System::Drawing::Size(234, 197);
			this->pct_scr_pp2->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_scr_pp2->TabIndex = 8;
			this->pct_scr_pp2->TabStop = false;
			// 
			// label21
			// 
			this->label21->AutoSize = true;
			this->label21->Location = System::Drawing::Point(262, 389);
			this->label21->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label21->Name = L"label21";
			this->label21->Size = System::Drawing::Size(40, 17);
			this->label21->TabIndex = 7;
			this->label21->Text = L"Value";
			this->label21->Click += gcnew System::EventHandler(this, &MainForm::label21_Click);
			// 
			// label20
			// 
			this->label20->AutoSize = true;
			this->label20->Location = System::Drawing::Point(403, 37);
			this->label20->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label20->Name = L"label20";
			this->label20->Size = System::Drawing::Size(93, 17);
			this->label20->TabIndex = 6;
			this->label20->Text = L"Source Image";
			// 
			// label19
			// 
			this->label19->AutoSize = true;
			this->label19->Location = System::Drawing::Point(407, 421);
			this->label19->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label19->Name = L"label19";
			this->label19->Size = System::Drawing::Size(90, 17);
			this->label19->TabIndex = 5;
			this->label19->Text = L"Target Image";
			// 
			// pct_trg_pp
			// 
			this->pct_trg_pp->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_trg_pp->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_trg_pp->Location = System::Drawing::Point(496, 421);
			this->pct_trg_pp->Margin = System::Windows::Forms::Padding(4);
			this->pct_trg_pp->Name = L"pct_trg_pp";
			this->pct_trg_pp->Size = System::Drawing::Size(700, 369);
			this->pct_trg_pp->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_trg_pp->TabIndex = 4;
			this->pct_trg_pp->TabStop = false;
			// 
			// pct_scr_pp
			// 
			this->pct_scr_pp->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_scr_pp->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_scr_pp->Location = System::Drawing::Point(496, 37);
			this->pct_scr_pp->Margin = System::Windows::Forms::Padding(4);
			this->pct_scr_pp->Name = L"pct_scr_pp";
			this->pct_scr_pp->Size = System::Drawing::Size(700, 369);
			this->pct_scr_pp->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_scr_pp->TabIndex = 3;
			this->pct_scr_pp->TabStop = false;
			// 
			// btn_lod_pp
			// 
			this->btn_lod_pp->Location = System::Drawing::Point(11, 452);
			this->btn_lod_pp->Margin = System::Windows::Forms::Padding(4);
			this->btn_lod_pp->Name = L"btn_lod_pp";
			this->btn_lod_pp->Size = System::Drawing::Size(220, 42);
			this->btn_lod_pp->TabIndex = 2;
			this->btn_lod_pp->Text = L"Load Image";
			this->btn_lod_pp->UseVisualStyleBackColor = true;
			this->btn_lod_pp->Click += gcnew System::EventHandler(this, &MainForm::btn_lod_pp_Click);
			// 
			// flowLayoutPanel2
			// 
			this->flowLayoutPanel2->BackColor = System::Drawing::Color::SteelBlue;
			this->flowLayoutPanel2->Controls->Add(this->btn_prt_adj);
			this->flowLayoutPanel2->Controls->Add(this->btn_blin);
			this->flowLayoutPanel2->Controls->Add(this->btn_hist);
			this->flowLayoutPanel2->Controls->Add(this->btn_log_trn);
			this->flowLayoutPanel2->Controls->Add(this->btn_pwr_low);
			this->flowLayoutPanel2->Controls->Add(this->btn_neg);
			this->flowLayoutPanel2->Location = System::Drawing::Point(8, 37);
			this->flowLayoutPanel2->Margin = System::Windows::Forms::Padding(4);
			this->flowLayoutPanel2->Name = L"flowLayoutPanel2";
			this->flowLayoutPanel2->Size = System::Drawing::Size(234, 325);
			this->flowLayoutPanel2->TabIndex = 1;
			// 
			// btn_prt_adj
			// 
			this->btn_prt_adj->Location = System::Drawing::Point(4, 4);
			this->btn_prt_adj->Margin = System::Windows::Forms::Padding(4);
			this->btn_prt_adj->Name = L"btn_prt_adj";
			this->btn_prt_adj->Size = System::Drawing::Size(220, 42);
			this->btn_prt_adj->TabIndex = 0;
			this->btn_prt_adj->Text = L"Brightness Adjustment";
			this->btn_prt_adj->UseVisualStyleBackColor = true;
			this->btn_prt_adj->Click += gcnew System::EventHandler(this, &MainForm::btn_prt_adj_Click);
			// 
			// btn_blin
			// 
			this->btn_blin->Location = System::Drawing::Point(4, 54);
			this->btn_blin->Margin = System::Windows::Forms::Padding(4);
			this->btn_blin->Name = L"btn_blin";
			this->btn_blin->Size = System::Drawing::Size(220, 42);
			this->btn_blin->TabIndex = 1;
			this->btn_blin->Text = L"Blinding";
			this->btn_blin->UseVisualStyleBackColor = true;
			this->btn_blin->Click += gcnew System::EventHandler(this, &MainForm::btn_blin_Click);
			// 
			// btn_hist
			// 
			this->btn_hist->Location = System::Drawing::Point(4, 104);
			this->btn_hist->Margin = System::Windows::Forms::Padding(4);
			this->btn_hist->Name = L"btn_hist";
			this->btn_hist->Size = System::Drawing::Size(220, 42);
			this->btn_hist->TabIndex = 2;
			this->btn_hist->Text = L"Histogram Equalization";
			this->btn_hist->UseVisualStyleBackColor = true;
			this->btn_hist->Click += gcnew System::EventHandler(this, &MainForm::btn_hist_Click);
			// 
			// btn_log_trn
			// 
			this->btn_log_trn->Location = System::Drawing::Point(4, 154);
			this->btn_log_trn->Margin = System::Windows::Forms::Padding(4);
			this->btn_log_trn->Name = L"btn_log_trn";
			this->btn_log_trn->Size = System::Drawing::Size(220, 42);
			this->btn_log_trn->TabIndex = 3;
			this->btn_log_trn->Text = L"Log Transformation";
			this->btn_log_trn->UseVisualStyleBackColor = true;
			this->btn_log_trn->Click += gcnew System::EventHandler(this, &MainForm::btn_log_trn_Click);
			// 
			// btn_pwr_low
			// 
			this->btn_pwr_low->Location = System::Drawing::Point(4, 204);
			this->btn_pwr_low->Margin = System::Windows::Forms::Padding(4);
			this->btn_pwr_low->Name = L"btn_pwr_low";
			this->btn_pwr_low->Size = System::Drawing::Size(220, 42);
			this->btn_pwr_low->TabIndex = 4;
			this->btn_pwr_low->Text = L"Power Law Transformation";
			this->btn_pwr_low->UseVisualStyleBackColor = true;
			this->btn_pwr_low->Click += gcnew System::EventHandler(this, &MainForm::btn_pwr_low_Click);
			// 
			// btn_neg
			// 
			this->btn_neg->Location = System::Drawing::Point(4, 254);
			this->btn_neg->Margin = System::Windows::Forms::Padding(4);
			this->btn_neg->Name = L"btn_neg";
			this->btn_neg->Size = System::Drawing::Size(220, 42);
			this->btn_neg->TabIndex = 5;
			this->btn_neg->Text = L"Negative";
			this->btn_neg->UseVisualStyleBackColor = true;
			this->btn_neg->Click += gcnew System::EventHandler(this, &MainForm::btn_neg_Click);
			// 
			// brt_adj_trackBar
			// 
			this->brt_adj_trackBar->BackColor = System::Drawing::Color::SteelBlue;
			this->brt_adj_trackBar->Location = System::Drawing::Point(8, 389);
			this->brt_adj_trackBar->Margin = System::Windows::Forms::Padding(4);
			this->brt_adj_trackBar->Maximum = 100;
			this->brt_adj_trackBar->Name = L"brt_adj_trackBar";
			this->brt_adj_trackBar->Size = System::Drawing::Size(234, 56);
			this->brt_adj_trackBar->TabIndex = 1;
			this->brt_adj_trackBar->TickFrequency = 5;
			this->brt_adj_trackBar->Scroll += gcnew System::EventHandler(this, &MainForm::brt_adj_trackBar_Scroll);
			// 
			// label18
			// 
			this->label18->AutoSize = true;
			this->label18->Location = System::Drawing::Point(7, 16);
			this->label18->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			this->label18->Name = L"label18";
			this->label18->Size = System::Drawing::Size(178, 17);
			this->label18->TabIndex = 0;
			this->label18->Text = L"Point Processing Operations";
			// 
			// freq_tab
			// 
			this->freq_tab->BackColor = System::Drawing::Color::CornflowerBlue;
			this->freq_tab->Controls->Add(this->label9);
			this->freq_tab->Controls->Add(this->textBox2);
			this->freq_tab->Controls->Add(this->over_write_freq);
			this->freq_tab->Controls->Add(this->label8);
			this->freq_tab->Controls->Add(this->pct_scr_fd);
			this->freq_tab->Controls->Add(this->label7);
			this->freq_tab->Controls->Add(this->label6);
			this->freq_tab->Controls->Add(this->label5);
			this->freq_tab->Controls->Add(this->label4);
			this->freq_tab->Controls->Add(this->label3);
			this->freq_tab->Controls->Add(this->pic_flt_ilp4);
			this->freq_tab->Controls->Add(this->pic_flt_aswp3);
			this->freq_tab->Controls->Add(this->pic_flt_bswp2);
			this->freq_tab->Controls->Add(this->pic_flt_IDFT5);
			this->freq_tab->Controls->Add(this->pic_flt_pad1);
			this->freq_tab->Controls->Add(this->label2);
			this->freq_tab->Controls->Add(this->frq_smo_flt_trackBar);
			this->freq_tab->Controls->Add(this->flowLayoutPanel4);
			this->freq_tab->Controls->Add(this->label1);
			this->freq_tab->Location = System::Drawing::Point(4, 25);
			this->freq_tab->Name = L"freq_tab";
			this->freq_tab->Padding = System::Windows::Forms::Padding(3);
			this->freq_tab->Size = System::Drawing::Size(1214, 797);
			this->freq_tab->TabIndex = 6;
			this->freq_tab->Text = L"Frequency Domain";
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Location = System::Drawing::Point(49, 338);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(40, 17);
			this->label9->TabIndex = 18;
			this->label9->Text = L"Value";
			// 
			// textBox2
			// 
			this->textBox2->Location = System::Drawing::Point(93, 337);
			this->textBox2->Name = L"textBox2";
			this->textBox2->Size = System::Drawing::Size(88, 24);
			this->textBox2->TabIndex = 17;
			// 
			// over_write_freq
			// 
			this->over_write_freq->AutoSize = true;
			this->over_write_freq->Location = System::Drawing::Point(7, 198);
			this->over_write_freq->Name = L"over_write_freq";
			this->over_write_freq->Size = System::Drawing::Size(104, 21);
			this->over_write_freq->TabIndex = 16;
			this->over_write_freq->Text = L"Over write \?";
			this->over_write_freq->UseVisualStyleBackColor = true;
			this->over_write_freq->CheckedChanged += gcnew System::EventHandler(this, &MainForm::over_write_freq_CheckedChanged);
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(7, 414);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(93, 17);
			this->label8->TabIndex = 15;
			this->label8->Text = L"Source Image";
			// 
			// pct_scr_fd
			// 
			this->pct_scr_fd->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_scr_fd->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_scr_fd->Location = System::Drawing::Point(7, 437);
			this->pct_scr_fd->Name = L"pct_scr_fd";
			this->pct_scr_fd->Size = System::Drawing::Size(226, 182);
			this->pct_scr_fd->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_scr_fd->TabIndex = 14;
			this->pct_scr_fd->TabStop = false;
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(648, 414);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(70, 17);
			this->label7->TabIndex = 13;
			this->label7->Text = L"After IDFT";
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(646, 7);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(54, 17);
			this->label6->TabIndex = 12;
			this->label6->Text = L"Padded";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(253, 536);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(96, 17);
			this->label5->TabIndex = 11;
			this->label5->Text = L"Ideal Low Pass";
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Location = System::Drawing::Point(253, 269);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(74, 17);
			this->label4->TabIndex = 10;
			this->label4->Text = L"After Swap";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Location = System::Drawing::Point(253, 7);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(77, 17);
			this->label3->TabIndex = 9;
			this->label3->Text = L"Befor Swap";
			// 
			// pic_flt_ilp4
			// 
			this->pic_flt_ilp4->BackColor = System::Drawing::Color::SteelBlue;
			this->pic_flt_ilp4->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pic_flt_ilp4->Location = System::Drawing::Point(253, 556);
			this->pic_flt_ilp4->Name = L"pic_flt_ilp4";
			this->pic_flt_ilp4->Size = System::Drawing::Size(388, 230);
			this->pic_flt_ilp4->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pic_flt_ilp4->TabIndex = 8;
			this->pic_flt_ilp4->TabStop = false;
			// 
			// pic_flt_aswp3
			// 
			this->pic_flt_aswp3->BackColor = System::Drawing::Color::SteelBlue;
			this->pic_flt_aswp3->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pic_flt_aswp3->Location = System::Drawing::Point(253, 293);
			this->pic_flt_aswp3->Name = L"pic_flt_aswp3";
			this->pic_flt_aswp3->Size = System::Drawing::Size(388, 230);
			this->pic_flt_aswp3->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pic_flt_aswp3->TabIndex = 7;
			this->pic_flt_aswp3->TabStop = false;
			// 
			// pic_flt_bswp2
			// 
			this->pic_flt_bswp2->BackColor = System::Drawing::Color::SteelBlue;
			this->pic_flt_bswp2->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pic_flt_bswp2->Location = System::Drawing::Point(253, 32);
			this->pic_flt_bswp2->Name = L"pic_flt_bswp2";
			this->pic_flt_bswp2->Size = System::Drawing::Size(388, 230);
			this->pic_flt_bswp2->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pic_flt_bswp2->TabIndex = 6;
			this->pic_flt_bswp2->TabStop = false;
			// 
			// pic_flt_IDFT5
			// 
			this->pic_flt_IDFT5->BackColor = System::Drawing::Color::SteelBlue;
			this->pic_flt_IDFT5->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pic_flt_IDFT5->Location = System::Drawing::Point(646, 437);
			this->pic_flt_IDFT5->Name = L"pic_flt_IDFT5";
			this->pic_flt_IDFT5->Size = System::Drawing::Size(563, 349);
			this->pic_flt_IDFT5->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pic_flt_IDFT5->TabIndex = 5;
			this->pic_flt_IDFT5->TabStop = false;
			// 
			// pic_flt_pad1
			// 
			this->pic_flt_pad1->BackColor = System::Drawing::Color::SteelBlue;
			this->pic_flt_pad1->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pic_flt_pad1->Location = System::Drawing::Point(646, 32);
			this->pic_flt_pad1->Name = L"pic_flt_pad1";
			this->pic_flt_pad1->Size = System::Drawing::Size(563, 349);
			this->pic_flt_pad1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pic_flt_pad1->TabIndex = 4;
			this->pic_flt_pad1->TabStop = false;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Location = System::Drawing::Point(10, 242);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(45, 17);
			this->label2->TabIndex = 3;
			this->label2->Text = L"Sigma";
			// 
			// frq_smo_flt_trackBar
			// 
			this->frq_smo_flt_trackBar->BackColor = System::Drawing::Color::SteelBlue;
			this->frq_smo_flt_trackBar->Location = System::Drawing::Point(7, 275);
			this->frq_smo_flt_trackBar->Maximum = 100;
			this->frq_smo_flt_trackBar->Name = L"frq_smo_flt_trackBar";
			this->frq_smo_flt_trackBar->Size = System::Drawing::Size(226, 56);
			this->frq_smo_flt_trackBar->TabIndex = 2;
			this->frq_smo_flt_trackBar->TickFrequency = 5;
			this->frq_smo_flt_trackBar->Scroll += gcnew System::EventHandler(this, &MainForm::frq_smo_flt_trackBar_Scroll);
			// 
			// flowLayoutPanel4
			// 
			this->flowLayoutPanel4->BackColor = System::Drawing::Color::SteelBlue;
			this->flowLayoutPanel4->Controls->Add(this->btn_frq_smo_flt);
			this->flowLayoutPanel4->Controls->Add(this->btn_frq_shr_flt);
			this->flowLayoutPanel4->Controls->Add(this->btn_lod_fd);
			this->flowLayoutPanel4->Location = System::Drawing::Point(7, 32);
			this->flowLayoutPanel4->Name = L"flowLayoutPanel4";
			this->flowLayoutPanel4->Size = System::Drawing::Size(226, 159);
			this->flowLayoutPanel4->TabIndex = 1;
			// 
			// btn_frq_smo_flt
			// 
			this->btn_frq_smo_flt->Location = System::Drawing::Point(3, 3);
			this->btn_frq_smo_flt->Name = L"btn_frq_smo_flt";
			this->btn_frq_smo_flt->Size = System::Drawing::Size(220, 42);
			this->btn_frq_smo_flt->TabIndex = 0;
			this->btn_frq_smo_flt->Text = L"Smoothing Filter";
			this->btn_frq_smo_flt->UseVisualStyleBackColor = true;
			this->btn_frq_smo_flt->Click += gcnew System::EventHandler(this, &MainForm::btn_frq_smo_flt_Click);
			// 
			// btn_frq_shr_flt
			// 
			this->btn_frq_shr_flt->Location = System::Drawing::Point(3, 51);
			this->btn_frq_shr_flt->Name = L"btn_frq_shr_flt";
			this->btn_frq_shr_flt->Size = System::Drawing::Size(220, 42);
			this->btn_frq_shr_flt->TabIndex = 1;
			this->btn_frq_shr_flt->Text = L"Sharpening Filter";
			this->btn_frq_shr_flt->UseVisualStyleBackColor = true;
			this->btn_frq_shr_flt->Click += gcnew System::EventHandler(this, &MainForm::btn_frq_shr_flt_Click);
			// 
			// btn_lod_fd
			// 
			this->btn_lod_fd->Location = System::Drawing::Point(3, 99);
			this->btn_lod_fd->Name = L"btn_lod_fd";
			this->btn_lod_fd->Size = System::Drawing::Size(220, 42);
			this->btn_lod_fd->TabIndex = 14;
			this->btn_lod_fd->Text = L"Load Image";
			this->btn_lod_fd->UseVisualStyleBackColor = true;
			this->btn_lod_fd->Click += gcnew System::EventHandler(this, &MainForm::btn_lod_fd_Click);
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(4, 12);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(162, 17);
			this->label1->TabIndex = 0;
			this->label1->Text = L"Frequency Domain Filters";
			// 
			// tabPage1
			// 
			this->tabPage1->BackColor = System::Drawing::Color::CornflowerBlue;
			this->tabPage1->Controls->Add(this->button1);
			this->tabPage1->Location = System::Drawing::Point(4, 25);
			this->tabPage1->Name = L"tabPage1";
			this->tabPage1->Padding = System::Windows::Forms::Padding(3);
			this->tabPage1->Size = System::Drawing::Size(1214, 797);
			this->tabPage1->TabIndex = 7;
			this->tabPage1->Text = L"Image Compression";
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(388, 176);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(311, 261);
			this->button1->TabIndex = 0;
			this->button1->Text = L"Save Last Over written Image";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MainForm::button1_Click_1);
			// 
			// tabPage3
			// 
			this->tabPage3->BackColor = System::Drawing::Color::CornflowerBlue;
			this->tabPage3->Controls->Add(this->label24);
			this->tabPage3->Controls->Add(this->label23);
			this->tabPage3->Controls->Add(this->pct_trg_seg);
			this->tabPage3->Controls->Add(this->pct_scr_seg);
			this->tabPage3->Controls->Add(this->over_write_seg);
			this->tabPage3->Controls->Add(this->flowLayoutPanel5);
			this->tabPage3->Controls->Add(this->label22);
			this->tabPage3->Location = System::Drawing::Point(4, 25);
			this->tabPage3->Name = L"tabPage3";
			this->tabPage3->Padding = System::Windows::Forms::Padding(3);
			this->tabPage3->Size = System::Drawing::Size(1214, 797);
			this->tabPage3->TabIndex = 8;
			this->tabPage3->Text = L"Image Segmentation";
			// 
			// label24
			// 
			this->label24->AutoSize = true;
			this->label24->Location = System::Drawing::Point(379, 404);
			this->label24->Name = L"label24";
			this->label24->Size = System::Drawing::Size(90, 17);
			this->label24->TabIndex = 5;
			this->label24->Text = L"Target Image";
			// 
			// label23
			// 
			this->label23->AutoSize = true;
			this->label23->Location = System::Drawing::Point(376, 7);
			this->label23->Name = L"label23";
			this->label23->Size = System::Drawing::Size(93, 17);
			this->label23->TabIndex = 4;
			this->label23->Text = L"Source Image";
			// 
			// pct_trg_seg
			// 
			this->pct_trg_seg->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_trg_seg->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_trg_seg->Location = System::Drawing::Point(379, 422);
			this->pct_trg_seg->Name = L"pct_trg_seg";
			this->pct_trg_seg->Size = System::Drawing::Size(700, 369);
			this->pct_trg_seg->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_trg_seg->TabIndex = 3;
			this->pct_trg_seg->TabStop = false;
			// 
			// pct_scr_seg
			// 
			this->pct_scr_seg->BackColor = System::Drawing::Color::SteelBlue;
			this->pct_scr_seg->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->pct_scr_seg->Location = System::Drawing::Point(376, 28);
			this->pct_scr_seg->Name = L"pct_scr_seg";
			this->pct_scr_seg->Size = System::Drawing::Size(700, 369);
			this->pct_scr_seg->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pct_scr_seg->TabIndex = 2;
			this->pct_scr_seg->TabStop = false;
			// 
			// over_write_seg
			// 
			this->over_write_seg->AutoSize = true;
			this->over_write_seg->Location = System::Drawing::Point(9, 288);
			this->over_write_seg->Name = L"over_write_seg";
			this->over_write_seg->Size = System::Drawing::Size(104, 21);
			this->over_write_seg->TabIndex = 4;
			this->over_write_seg->Text = L"Over write \?";
			this->over_write_seg->UseVisualStyleBackColor = true;
			this->over_write_seg->CheckedChanged += gcnew System::EventHandler(this, &MainForm::over_write_seg_CheckedChanged);
			// 
			// flowLayoutPanel5
			// 
			this->flowLayoutPanel5->BackColor = System::Drawing::Color::SteelBlue;
			this->flowLayoutPanel5->Controls->Add(this->btn_seg);
			this->flowLayoutPanel5->Controls->Add(this->btn_lod_seg);
			this->flowLayoutPanel5->Controls->Add(this->label25);
			this->flowLayoutPanel5->Controls->Add(this->thr_adj_trackBar);
			this->flowLayoutPanel5->Controls->Add(this->label26);
			this->flowLayoutPanel5->Controls->Add(this->textBox3);
			this->flowLayoutPanel5->Location = System::Drawing::Point(9, 28);
			this->flowLayoutPanel5->Name = L"flowLayoutPanel5";
			this->flowLayoutPanel5->Size = System::Drawing::Size(226, 254);
			this->flowLayoutPanel5->TabIndex = 1;
			// 
			// btn_seg
			// 
			this->btn_seg->Location = System::Drawing::Point(3, 3);
			this->btn_seg->Name = L"btn_seg";
			this->btn_seg->Size = System::Drawing::Size(220, 42);
			this->btn_seg->TabIndex = 0;
			this->btn_seg->Text = L"Segmentation";
			this->btn_seg->UseVisualStyleBackColor = true;
			this->btn_seg->Click += gcnew System::EventHandler(this, &MainForm::btn_seg_Click);
			// 
			// btn_lod_seg
			// 
			this->btn_lod_seg->Location = System::Drawing::Point(3, 51);
			this->btn_lod_seg->Name = L"btn_lod_seg";
			this->btn_lod_seg->Size = System::Drawing::Size(220, 42);
			this->btn_lod_seg->TabIndex = 1;
			this->btn_lod_seg->Text = L"Load Image";
			this->btn_lod_seg->UseVisualStyleBackColor = true;
			this->btn_lod_seg->Click += gcnew System::EventHandler(this, &MainForm::btn_lod_seg_Click);
			// 
			// label25
			// 
			this->label25->AutoSize = true;
			this->label25->Location = System::Drawing::Point(3, 96);
			this->label25->Name = L"label25";
			this->label25->Size = System::Drawing::Size(68, 17);
			this->label25->TabIndex = 3;
			this->label25->Text = L"Threshold";
			// 
			// thr_adj_trackBar
			// 
			this->thr_adj_trackBar->BackColor = System::Drawing::Color::CornflowerBlue;
			this->thr_adj_trackBar->Location = System::Drawing::Point(3, 116);
			this->thr_adj_trackBar->Maximum = 255;
			this->thr_adj_trackBar->Name = L"thr_adj_trackBar";
			this->thr_adj_trackBar->Size = System::Drawing::Size(220, 56);
			this->thr_adj_trackBar->TabIndex = 2;
			this->thr_adj_trackBar->TickFrequency = 15;
			this->thr_adj_trackBar->Scroll += gcnew System::EventHandler(this, &MainForm::thr_adj_trackBar_Scroll);
			// 
			// label26
			// 
			this->label26->AutoSize = true;
			this->label26->Location = System::Drawing::Point(3, 175);
			this->label26->Name = L"label26";
			this->label26->Size = System::Drawing::Size(104, 17);
			this->label26->TabIndex = 5;
			this->label26->Text = L"Threshold Value";
			// 
			// textBox3
			// 
			this->textBox3->Location = System::Drawing::Point(113, 178);
			this->textBox3->Name = L"textBox3";
			this->textBox3->Size = System::Drawing::Size(88, 24);
			this->textBox3->TabIndex = 6;
			// 
			// label22
			// 
			this->label22->AutoSize = true;
			this->label22->Location = System::Drawing::Point(6, 7);
			this->label22->Name = L"label22";
			this->label22->Size = System::Drawing::Size(93, 17);
			this->label22->TabIndex = 0;
			this->label22->Text = L"Segmentation";
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(7, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::SystemColors::ActiveCaption;
			this->ClientSize = System::Drawing::Size(1225, 827);
			this->Controls->Add(this->btnBrowse3);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::Fixed3D;
			this->Icon = (cli::safe_cast<System::Drawing::Icon^>(resources->GetObject(L"$this.Icon")));
			this->Margin = System::Windows::Forms::Padding(4);
			this->Name = L"MainForm";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->Text = L"Kamal Tool";
			this->btnBrowse3->ResumeLayout(false);
			this->tabPage2->ResumeLayout(false);
			this->tabPage2->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->rot_adj_trackBar))->EndInit();
			this->flowLayoutPanel3->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_tar_op))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_op))->EndInit();
			this->tabPage4->ResumeLayout(false);
			this->tabPage4->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_tar_filter))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_filter))->EndInit();
			this->flowLayoutPanel1->ResumeLayout(false);
			this->flowLayoutPanel1->PerformLayout();
			this->tabPage5->ResumeLayout(false);
			this->tabPage5->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_pp2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_trg_pp))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_pp))->EndInit();
			this->flowLayoutPanel2->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->brt_adj_trackBar))->EndInit();
			this->freq_tab->ResumeLayout(false);
			this->freq_tab->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_fd))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_ilp4))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_aswp3))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_bswp2))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_IDFT5))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pic_flt_pad1))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->frq_smo_flt_trackBar))->EndInit();
			this->flowLayoutPanel4->ResumeLayout(false);
			this->tabPage1->ResumeLayout(false);
			this->tabPage3->ResumeLayout(false);
			this->tabPage3->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_trg_seg))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pct_scr_seg))->EndInit();
			this->flowLayoutPanel5->ResumeLayout(false);
			this->flowLayoutPanel5->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->thr_adj_trackBar))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion

//Now we going to create all fiter method 

	private: cv::Mat HistBalance(Mat inputMat) {
		Mat hsv, disp;
		cvtColor(inputMat, hsv, CV_BGR2HSV);  // Convert BGR to HSV
		vector<Mat> hsv_channels;
		split(hsv, hsv_channels);				// Get the V channel
		equalizeHist(hsv_channels[2], hsv_channels[2]);		// Balance V Channel
		merge(hsv_channels, hsv); // merge V channel into hsv image
		cvtColor(hsv, disp, CV_HSV2BGR); // Covert back to BGR image
		convertScaleAbs(disp, disp);
		return disp;
	 }
	private: cv::Mat BlurFilter(Mat inputMat) {
		Mat dst;
		blur(inputMat, dst, cv::Size(3, 3));
		return dst;
	}
	private: cv::Mat SobelFilter(Mat inputMat) {
		Mat dst;
		Mat dst2;
		Sobel(inputMat, dst, CV_64F, 1, 1);
		convertScaleAbs(dst, dst2);
		return dst2;
	}
	private: cv::Mat LaplFilter(Mat inputMat) {
		Mat dst,dst2;
		Laplacian(inputMat, dst, CV_64F);
		convertScaleAbs(dst, dst2);
		return dst2;
	}
	private: cv::Mat CannyFilter(Mat inputMat) {
		Mat dst, dst1, dst2;
		cvtColor(inputMat, dst, CV_BGR2GRAY);
		GaussianBlur(dst, dst, cv::Size(9, 9), 2);
		double t1 = 30, t2 = 200;
		Canny(dst, dst1, t1, t2, 3, false);
		convertScaleAbs(dst1, dst2);
		return dst2;
	}
	private: cv::Mat CuzFilter(Mat inputMat, double m[3][3]) {
		Mat dst, dst2;
		Mat M = Mat(3, 3, CV_64F, m);
		cv::filter2D(inputMat, dst, inputMat.depth(), M);
		convertScaleAbs(dst, dst2);
		return dst2;
	}
	private: Mat ToResize(Mat src) {	// Rezise image method
	Mat dst;
	resize(src, dst, cv::Size(320, 240), 0, 0, 1); // we need to define new size onn cv:Sizze(width, height)
	return dst;
	
}
	private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
		// Here is detect line/cirlce code you can download code and check for more detail
		Mat src;
		OpenFileDialog^ opDialog = gcnew OpenFileDialog();
		opDialog->Filter = "Image(*.bmp; *.jpg)|*.bmp;*.jpg|All files (*.*)|*.*||";
		if (opDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
		{
			return;
		}
		Bitmap^ bmpSrc = gcnew Bitmap(opDialog->FileName);
		//ptbSrc2->Image = bmpSrc;
		//ptbSrc2->Refresh();
		str2char str2ch; // convert string use jackylib
						 // load image into src variable and show as OpenCV method
		resize(imread(str2ch.ConvertString2Char(opDialog->FileName)),src,cv::Size(200,200));
		Mat gray;
		cvtColor(src, gray, CV_BGR2GRAY);
		GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
		// Find line
		Mat canny;
		Canny(gray, canny, 100, 200, 3, false);
		vector<Vec4i> lines;
		HoughLinesP(canny, lines, 1, CV_PI / 180, 50, 60, 10);
		// Find circle
		vector<Vec3f> circles;
		HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, 100, 200, 100, 0, 0);
		// Draw cirle and line on image
		for (int i = 0; i < lines.size(); i++)
		{
			Vec4i l = lines[i];
			line(src, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), Scalar(0, 0, 255), 2);
		}

		for (int i = 0; i < circles.size(); i++)
		{
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]); 
			circle(src, center, radius, Scalar(0, 0, 255), 2, 8, 0);
		}
		
		mat2picture bmpconvert;
		Mat display;
		convertScaleAbs(src, display);
		//ptbDetect->Image = bmpconvert.Mat2Bimap(display);
		//ptbDetect->Refresh();

	}
private: System::Void btnBrowse3_SelectedIndexChanged(System::Object^  sender, System::EventArgs^  e) {
	
}
private: System::Void tabPage3_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void button7_Click(System::Object^ sender, System::EventArgs^ e) 
{
	Mat src;
	OpenFileDialog^ opDialog = gcnew OpenFileDialog();
	opDialog->Filter = "Image(*.bmp; *.jpg)|*.bmp;*.jpg|All files (*.*)|*.*||";
	if (opDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
	{
		return;
	}
	Bitmap^ bmpSrc = gcnew Bitmap(opDialog->FileName);
	str2char str2ch;
	src = ToResize(imread(str2ch.ConvertString2Char(opDialog->FileName)));
	output_mat = src;
	public_matrix = src;
	pct_scr_op->Image = bmpSrc;
	pct_scr_op->Refresh();
}
	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e)
	{
		mat2picture bimapconvert;
		Mat mat_of_function;
		mat_of_function = ToResize(output_mat);
		if (over_write_flag1 == 1) {
			mat_of_function = image_flipping(public_matrix);
			
		}
		else {
			mat_of_function = image_flipping(output_mat);
		}
		public_matrix = mat_of_function;
	pct_tar_op->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_op->Refresh();
}
private: System::Void button3_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag1 == 1) {
		mat_of_function = rotation90(public_matrix);

	}
	else {
		mat_of_function = rotation90(output_mat);
	}
	public_matrix = mat_of_function;
	pct_tar_op->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_op->Refresh();
}
private: System::Void button4_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag1 == 1) {
		mat_of_function = rotation180(public_matrix);

	}
	else {
		mat_of_function = rotation180(output_mat);
	}
	public_matrix = mat_of_function;
	pct_tar_op->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_op->Refresh();
}
private: System::Void btn_avg_filter_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag2 == 1) {
		mat_of_function = smoothing_filter(public_matrix);
		
	}
	else {
		mat_of_function = smoothing_filter(output_mat);
	}
	public_matrix = mat_of_function;
	pct_tar_filter->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_filter->Refresh();
}
private: System::Void btn_lod_filters_Click(System::Object^ sender, System::EventArgs^ e) 
{
	Mat src;
	OpenFileDialog^ opDialog = gcnew OpenFileDialog();
	opDialog->Filter = "Image(*.bmp; *.jpg)|*.bmp;*.jpg|All files (*.*)|*.*||";
	if (opDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
	{
		return;
	}
	Bitmap^ bmpSrc = gcnew Bitmap(opDialog->FileName);
	str2char str2ch;
	src = ToResize(imread(str2ch.ConvertString2Char(opDialog->FileName)));
	output_mat = src;
	public_matrix = src;
	pct_scr_filter->Image = bmpSrc;
	pct_scr_filter->Refresh();
}
private: System::Void btn_mid_filter_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag2 == 1) {
		mat_of_function = median_filter(public_matrix);

	}
	else {
		mat_of_function = median_filter(output_mat);
	}
	public_matrix = mat_of_function;
	pct_tar_filter->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_filter->Refresh();
}
private: System::Void btc_pre_filter_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag2 == 1) {
		mat_of_function = prewitt_sharpening(public_matrix);

	}
	else {
		mat_of_function = prewitt_sharpening(output_mat);
	}
	public_matrix = mat_of_function;
	pct_tar_filter->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_filter->Refresh();
}
private: System::Void btc_sob_filter_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag2 == 1) {
		mat_of_function = sobel_edge_detection(public_matrix);

	}
	else {
		mat_of_function = sobel_edge_detection(output_mat);
	}
	public_matrix = mat_of_function;
	pct_tar_filter->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_filter->Refresh();
}
private: System::Void label16_Click(System::Object^ sender, System::EventArgs^ e) 
{

}
private: System::Void btn_prt_adj_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag3 == 1) {
		if (slidervalue >= 50)
		{
			mat_of_function = Bright_Inc(public_matrix);
		}
		else
		{
			mat_of_function = Bright_Dec(public_matrix);
		}
	}
	else {
		if (slidervalue >= 50)
		{
			mat_of_function = Bright_Inc(output_mat);
		}
		else
		{
			mat_of_function = Bright_Dec(output_mat);
		}

	}
	public_matrix = mat_of_function;
	pct_trg_pp->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_trg_pp->Refresh();
}
private: System::Void brt_adj_trackBar_Scroll(System::Object^ sender, System::EventArgs^ e) 
{
	slidervalue = brt_adj_trackBar->Value;
	textBox1->Text = slidervalue.ToString("N2");
}
private: System::Void btn_lod_pp_Click(System::Object^ sender, System::EventArgs^ e) 
{
	Mat src;
	OpenFileDialog^ opDialog = gcnew OpenFileDialog();
	opDialog->Filter = "Image(*.bmp; *.jpg)|*.bmp;*.jpg|All files (*.*)|*.*||";
	if (opDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
	{
		return;
	}
	Bitmap^ bmpSrc = gcnew Bitmap(opDialog->FileName);
	str2char str2ch;
	src = ToResize(imread(str2ch.ConvertString2Char(opDialog->FileName)));
	output_mat = src;
	public_matrix = src;
	pct_scr_pp->Image = bmpSrc;
	pct_scr_pp->Refresh();
}
private: System::Void btn_blin_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	mat_of_function = Image_blinding(output_mat,output_mat2);
	 pct_trg_pp->Image = bimapconvert.Mat2Bimap(mat_of_function);
	 pct_trg_pp->Refresh();
}
private: System::Void btn_lod_pp2_Click(System::Object^ sender, System::EventArgs^ e) 
{
	Mat src;
	OpenFileDialog^ opDialog = gcnew OpenFileDialog();
	opDialog->Filter = "Image(*.bmp; *.jpg)|*.bmp;*.jpg|All files (*.*)|*.*||";
	if (opDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
	{
		return;
	}
	Bitmap^ bmpSrc = gcnew Bitmap(opDialog->FileName);
	str2char str2ch;
	src = ToResize(imread(str2ch.ConvertString2Char(opDialog->FileName)));
	output_mat2 = src;
	pct_scr_pp2->Image = bmpSrc;
	pct_scr_pp2->Refresh();
}
private: System::Void btn_log_trn_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag3 == 1)
	{
		mat_of_function = Log_Transformation(public_matrix , slidervalue);
	}
	else
	{
		mat_of_function = Log_Transformation(public_matrix , slidervalue);
	}
	public_matrix = mat_of_function;
	pct_trg_pp->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_trg_pp->Refresh();
}
private: System::Void btn_pwr_low_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag3 == 1)
	{
		mat_of_function = Power_Transformation(public_matrix, slidervalue);
	}
	else
	{
		mat_of_function = Power_Transformation(public_matrix, slidervalue);
	}
	public_matrix = mat_of_function;
	pct_trg_pp->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_trg_pp->Refresh();
}
private: System::Void btn_neg_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(public_matrix);
	if (over_write_flag3 == 1)
	{
		mat_of_function = Negative(public_matrix);
	}
	else
	{
		mat_of_function = Negative(output_mat);
	}
	public_matrix = mat_of_function;
	pct_trg_pp->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_trg_pp->Refresh();
}
private: System::Void label21_Click(System::Object^ sender, System::EventArgs^ e) 
{
	//label21->Text = Convert::ToString(slidervalue);
}
private: System::Void btn_hist_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (slidervalue >= 50)
	{
		mat_of_function = histogram(public_matrix);
	}
	else
	{
		mat_of_function = histogram(public_matrix);
	}
	pct_trg_pp->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_trg_pp->Refresh();
}
private: System::Void btn_zoom_op_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag1 == 1) {
		mat_of_function = zooming(public_matrix);
		
	}
	else {
		mat_of_function = zooming(output_mat);

	}
	public_matrix = mat_of_function;
	pct_tar_op->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_op->Refresh();
}
private: System::Void overwrite_box_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	over_write_flag1 = overwrite_box->Checked;
}
private: System::Void over_write_filter_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	over_write_flag2 = over_write_filter->Checked;
}
private: System::Void btn_rot_270_op_Click(System::Object^ sender, System::EventArgs^ e) {
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag1 == 1) {
		mat_of_function = rotation270(public_matrix);

	}
	else {
		mat_of_function = rotation270(output_mat);
	}
	public_matrix = mat_of_function;
	pct_tar_op->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_op->Refresh();
}
private: System::Void btn_frq_smo_flt_Click(System::Object^ sender, System::EventArgs^ e) 
{
	 //= Frequency_Domain_Smoothing_Filter();
	 mat2picture bimapconvert;
	 Mat mat_of_function0, mat_of_function1, mat_of_function2, mat_of_function3, mat_of_function4;
	 mat_of_function0 = ToResize(output_mat); mat_of_function1 = ToResize(output_mat); mat_of_function2 = ToResize(output_mat);
	 mat_of_function3 = ToResize(output_mat); mat_of_function4 = ToResize(output_mat);
	 if (over_write_flag_flt == 1) 
	 {
		 mat_of_function0 = Frequency_Domain_Smoothing_Filter(public_matrix,1,sigma_glo);
		 pic_flt_pad1->Image = bimapconvert.Mat2Bimap(mat_of_function0);
		 pic_flt_pad1->Refresh();
		 //--------------------------------------------------------------------------
		 mat_of_function1 = Frequency_Domain_Smoothing_Filter(public_matrix, 2, sigma_glo);
		 pic_flt_bswp2->Image = bimapconvert.Mat2Bimap(mat_of_function1);
		 pic_flt_bswp2->Refresh();
		 //--------------------------------------------------------------------------
		 mat_of_function2 = Frequency_Domain_Smoothing_Filter(public_matrix, 3, sigma_glo);
		 pic_flt_aswp3->Image = bimapconvert.Mat2Bimap(mat_of_function2);
		 pic_flt_aswp3->Refresh();
		 //--------------------------------------------------------------------------
		 mat_of_function3 = Frequency_Domain_Smoothing_Filter(public_matrix, 4, sigma_glo);
		 pic_flt_ilp4->Image = bimapconvert.Mat2Bimap(mat_of_function3);
		 pic_flt_ilp4->Refresh();
		 //--------------------------------------------------------------------------
		 mat_of_function4 = Frequency_Domain_Smoothing_Filter(public_matrix, 5, sigma_glo);
		 public_matrix = mat_of_function4;
		 pic_flt_IDFT5->Image = bimapconvert.Mat2Bimap(mat_of_function4);
		 pic_flt_IDFT5->Refresh();
	 }
	 else 
	 {
		 mat_of_function0 = Frequency_Domain_Smoothing_Filter(output_mat, 1, sigma_glo);
		 pic_flt_pad1->Image = bimapconvert.Mat2Bimap(mat_of_function0);
		 pic_flt_pad1->Refresh();
		 //--------------------------------------------------------------------------
		 mat_of_function1 = Frequency_Domain_Smoothing_Filter(output_mat, 2, sigma_glo);
		 pic_flt_bswp2->Image = bimapconvert.Mat2Bimap(mat_of_function1);
		 pic_flt_bswp2->Refresh();
		 //--------------------------------------------------------------------------
		 mat_of_function2 = Frequency_Domain_Smoothing_Filter(output_mat, 3, sigma_glo);
		 pic_flt_aswp3->Image = bimapconvert.Mat2Bimap(mat_of_function2);
		 pic_flt_aswp3->Refresh();
		 //--------------------------------------------------------------------------
		 mat_of_function3 = Frequency_Domain_Smoothing_Filter(output_mat, 4, sigma_glo);
		 pic_flt_ilp4->Image = bimapconvert.Mat2Bimap(mat_of_function3);
		 pic_flt_ilp4->Refresh();
		 //--------------------------------------------------------------------------
		 mat_of_function4 = Frequency_Domain_Smoothing_Filter(output_mat, 5, sigma_glo);
		 pic_flt_IDFT5->Image = bimapconvert.Mat2Bimap(mat_of_function4);
		 pic_flt_IDFT5->Refresh();
	 }
	 // over_write_flag_flt
}
private: System::Void btn_lod_fd_Click(System::Object^ sender, System::EventArgs^ e) 
{
	Mat src;
	OpenFileDialog^ opDialog = gcnew OpenFileDialog();
	opDialog->Filter = "Image(*.bmp; *.jpg)|*.bmp;*.jpg|All files (*.*)|*.*||";
	if (opDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
	{
		return;
	}
	Bitmap^ bmpSrc = gcnew Bitmap(opDialog->FileName);
	str2char str2ch;
	src = ToResize(imread(str2ch.ConvertString2Char(opDialog->FileName)));
	output_mat = src;
	public_matrix = src;
	pct_scr_fd->Image = bmpSrc;
	pct_scr_fd->Refresh();
}
private: System::Void over_write_freq_CheckedChanged(System::Object^ sender, System::EventArgs^ e) 
{
	over_write_flag_flt = over_write_freq->Checked;
}
private: System::Void frq_smo_flt_trackBar_Scroll(System::Object^ sender, System::EventArgs^ e) 
{
	sigma_glo = frq_smo_flt_trackBar->Value;
	textBox2->Text = sigma_glo.ToString("N2");
}
private: System::Void btn_lod_seg_Click(System::Object^ sender, System::EventArgs^ e) 
{
	Mat src;
	OpenFileDialog^ opDialog = gcnew OpenFileDialog();
	opDialog->Filter = "Image(*.bmp; *.jpg)|*.bmp;*.jpg|All files (*.*)|*.*||";
	if (opDialog->ShowDialog() == System::Windows::Forms::DialogResult::Cancel)
	{
		return;
	}
	Bitmap^ bmpSrc = gcnew Bitmap(opDialog->FileName);
	str2char str2ch;
	src = ToResize(imread(str2ch.ConvertString2Char(opDialog->FileName)));
	output_mat = src;
	public_matrix = src;
	pct_scr_seg->Image = bmpSrc;
	pct_scr_seg->Refresh();
}
private: System::Void btn_seg_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag_seg == 1) {
		mat_of_function = segmentation(public_matrix, threshold_gol);

	}
	else {
		mat_of_function = segmentation(output_mat, threshold_gol);
	}
	public_matrix = mat_of_function;
	pct_trg_seg->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_trg_seg->Refresh();
}
private: System::Void over_write_seg_CheckedChanged(System::Object^ sender, System::EventArgs^ e) 
{
	over_write_flag_seg = over_write_seg->Checked;
}
private: System::Void thr_adj_trackBar_Scroll(System::Object^ sender, System::EventArgs^ e) 
{
	threshold_gol = thr_adj_trackBar->Value;
	textBox3->Text = threshold_gol.ToString("N2");
}
private: System::Void rot_adj_trackBar_Scroll(System::Object^ sender, System::EventArgs^ e) 
{
	slidervalue_rot = rot_adj_trackBar->Value;
	textBox4->Text = slidervalue_rot.ToString("N2");
}
private: System::Void btn_rot_rnm_op_Click(System::Object^ sender, System::EventArgs^ e) 
{
	mat2picture bimapconvert;
	Mat mat_of_function;
	mat_of_function = ToResize(output_mat);
	if (over_write_flag1 == 1) {
		mat_of_function = rotat_any_ang(public_matrix , slidervalue_rot , scale_gol);
	}
	else {
		mat_of_function = rotat_any_ang(output_mat, slidervalue_rot, scale_gol);
	}
	public_matrix = mat_of_function;
	pct_tar_op->Image = bimapconvert.Mat2Bimap(mat_of_function);
	pct_tar_op->Refresh();
}
private: System::Void textBox4_TextChanged(System::Object^ sender, System::EventArgs^ e) 
{

}
private: System::Void textBox5_TextChanged(System::Object^ sender, System::EventArgs^ e) 
{
	scale_gol = double::Parse(textBox5->Text);
}
private: System::Void checkBox1_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	over_write_flag3 = checkBox7->Checked;
}
private: System::Void button1_Click_1(System::Object^ sender, System::EventArgs^ e) {
	
	huffman_encryption(public_matrix, "output_huffman_encrypted.txt");

}
private: System::Void btn_frq_shr_flt_Click(System::Object^ sender, System::EventArgs^ e) 
{

	//= Frequency_Domain_Smoothing_Filter();
	mat2picture bimapconvert;
	Mat mat_of_function0, mat_of_function1, mat_of_function2, mat_of_function3, mat_of_function4;
	mat_of_function0 = ToResize(output_mat); mat_of_function1 = ToResize(output_mat); mat_of_function2 = ToResize(output_mat);
	mat_of_function3 = ToResize(output_mat); mat_of_function4 = ToResize(output_mat);
	if (over_write_flag_flt == 1)
	{
		mat_of_function0 = Frequency_Domain_Sharpening_Filter(public_matrix, 1, sigma_glo);
		pic_flt_pad1->Image = bimapconvert.Mat2Bimap(mat_of_function0);
		pic_flt_pad1->Refresh();
		//--------------------------------------------------------------------------
		mat_of_function1 = Frequency_Domain_Sharpening_Filter(public_matrix, 2, sigma_glo);
		pic_flt_bswp2->Image = bimapconvert.Mat2Bimap(mat_of_function1);
		pic_flt_bswp2->Refresh();
		//--------------------------------------------------------------------------
		mat_of_function2 = Frequency_Domain_Sharpening_Filter(public_matrix, 3, sigma_glo);
		pic_flt_aswp3->Image = bimapconvert.Mat2Bimap(mat_of_function2);
		pic_flt_aswp3->Refresh();
		//--------------------------------------------------------------------------
		mat_of_function3 = Frequency_Domain_Sharpening_Filter(public_matrix, 4, sigma_glo);
		pic_flt_ilp4->Image = bimapconvert.Mat2Bimap(mat_of_function3);
		pic_flt_ilp4->Refresh();
		//--------------------------------------------------------------------------
		mat_of_function4 = Frequency_Domain_Sharpening_Filter(public_matrix, 5, sigma_glo);
		public_matrix = mat_of_function4;
		pic_flt_IDFT5->Image = bimapconvert.Mat2Bimap(mat_of_function4);
		pic_flt_IDFT5->Refresh();
	}
	else
	{
		mat_of_function0 = Frequency_Domain_Sharpening_Filter(output_mat, 1, sigma_glo);
		pic_flt_pad1->Image = bimapconvert.Mat2Bimap(mat_of_function0);
		pic_flt_pad1->Refresh();
		//--------------------------------------------------------------------------
		mat_of_function1 = Frequency_Domain_Sharpening_Filter(output_mat, 2, sigma_glo);
		pic_flt_bswp2->Image = bimapconvert.Mat2Bimap(mat_of_function1);
		pic_flt_bswp2->Refresh();
		//--------------------------------------------------------------------------
		mat_of_function2 = Frequency_Domain_Sharpening_Filter(output_mat, 3, sigma_glo);
		pic_flt_aswp3->Image = bimapconvert.Mat2Bimap(mat_of_function2);
		pic_flt_aswp3->Refresh();
		//--------------------------------------------------------------------------
		mat_of_function3 = Frequency_Domain_Sharpening_Filter(output_mat, 4, sigma_glo);
		pic_flt_ilp4->Image = bimapconvert.Mat2Bimap(mat_of_function3);
		pic_flt_ilp4->Refresh();
		//--------------------------------------------------------------------------
		mat_of_function4 = Frequency_Domain_Sharpening_Filter(output_mat, 5, sigma_glo);
		pic_flt_IDFT5->Image = bimapconvert.Mat2Bimap(mat_of_function4);
		pic_flt_IDFT5->Refresh();
	}
	// over_write_flag_flt

}
};
}
