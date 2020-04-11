
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

/// 全局变量
char* image_window = (char*)"Source Image";
char* result_window = (char*)"Result window";
const float rate_row = 0.6;
const float rate_col = 0.1;
const float match_rate = 0.5;

const string img_number = "9";

const string str = "D:\\User\\Desktop\\毕业设计\\image\\image edit\\";
const string strLeft = "\\Left_";
const string strRight = "\\Right_";
const string strJPG = ".jpg";
const string strTemplate = "\\template.jpg";
const string strResult = "\\result.jpg";

int match_method = 0;
//0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED


/// 函数声明
Point MatchingMethod(Mat img, Mat templ);

/** @主函数 */
int main(int argc, char** argv)
{
	/// 载入原图像和模板块
	Mat img1, img2;
	//img1 = imread("D:\\User\\Desktop\\毕业设计\\image\\image edit\\5\\Left_5SR.jpg", 0);
	//img2 = imread("D:\\User\\Desktop\\毕业设计\\image\\image edit\\5\\Right_5SR.jpg", 0);

	img1 = imread((str + img_number + strLeft + img_number + strJPG).data(), 0);
	img2 = imread((str + img_number + strRight + img_number + strJPG).data(), 0);

	

	if (img1.empty() || img2.empty() )
	{
		cout << "图像加载失败" << endl;
		return -1;
	}

	flip(img2, img2, 1);
	Mat img2_half(img2.rows, (int)img2.cols * match_rate, CV_8UC1, Scalar(255));
	for (int i = 0; i < img2_half.rows; i++)
	{
		for (int j = 0; j < img2_half.cols; j++)
		{
			img2_half.at<uchar>(i, j) = img2.at<uchar>(i, j);
		}
	}
	imwrite((str + img_number + (string)"\\Flipped.jpg").data(), img2);
	
	//取img1 右侧中间，大小为 rows * rate_row , cols * rate_col 作为模板
	Mat templ((int)img1.rows * rate_row, (int)img1.cols * rate_col, CV_8UC1, Scalar(255));

	for (int i = 0; i < templ.rows; i++)
	{
		for (int j = 0; j < templ.cols; j++)
		{
			templ.at<uchar>(i, j) = img1.at<uchar>(i+(int)img1.rows * (1 - rate_row) / 2, j + (int)img1.cols * (1-rate_col));
		}
	}

	imwrite((str + img_number + strTemplate).data(), templ);

	//计算匹配点在原图下的坐标
	Point origin(MatchingMethod(img1, templ));
	Point target(MatchingMethod(img2_half, templ));

	//创建结果图像
	Mat StitchResult(img1.rows + img2.rows * 2, img1.cols + img2.cols, CV_8UC1, Scalar(255));

	//把img1放到左侧中间，即原图放在左上角并向下平移img2.rows
	for (int i = 0; i < img1.rows; i++)
	{
		for (int j = 0; j < img1.cols; j++) 
		{
			StitchResult.at<uchar>(i+img2.rows, j) = img1.at<uchar>(i, j); //注意反序
		}
	}

	//img1匹配点在结果图内坐标为 (origin.x, origin.y + img2.rows)
	origin.y = origin.y + img2.rows;

	//拷贝img2 全部图像，img2 在结果图内坐标为 (i,j) + origin(x,y) - target(x,y)
	for (int i = 0;i < img2.rows; i++)
	{
		for (int j = 0; j < img2.cols; j++)
		{
			StitchResult.at<uchar>(i + origin.y - target.y, j + origin.x - target.x ) = img2.at<uchar>(i, j);
		}
	}

	//绘制出模板
	//rectangle(StitchResult, origin, Point(origin.x + templ.cols, origin.y + templ.rows), Scalar::all(0), 2, 8, 0);

	//imshow(result_window, StitchResult);
	imwrite((str + img_number + strResult).data(), StitchResult);



	waitKey(0);
	return 0;
}

//模板匹配函数
Point MatchingMethod(Mat img, Mat templ)
{
	/// 将被显示的原图像
	Mat img_display;
	Mat result;
	img.copyTo(img_display);

	/// 创建输出结果的矩阵
	int result_cols = img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create(result_cols, result_rows, CV_32FC1);

	/// 进行匹配和标准化
	
	matchTemplate(img, templ, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	/// 通过函数 minMaxLoc 定位最匹配的位置
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	/// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	/// 让我看看您的最终结果
	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);

	//imshow(image_window, img_display);
	//imshow(result_window, result);

	cout << "Point:(" << matchLoc.x << "," << matchLoc.y << ")" << endl;

	//result = result * 255; //return to 0-255
	imwrite((str + img_number + (string)"\\Temp.jpg").data(), img_display);

	return matchLoc;
}

