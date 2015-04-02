#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>  
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <direct.h>
#include <fstream>
#include "ml.h"
#include <utility>
#include <list>
#include <io.h>
#include "SBE.h"

#include "lbp.hpp"

#define RANDPROBABILITY 5
#define PIXELBACKGROUND	
#define REINITIALRATIO 0.8	
#define REINITIALFRAMENUM 1	
#define RECORDBACKGROUND
//#define CANCELCANNY		
//#define REPAIR			

using namespace std;
using namespace cv;

void loadImages_gary(string path, vector<IplImage*> &images);
void loadImages(string path, vector<IplImage*> &images, vector<IplImage*> &gaus);	
void loadVideo(string path, vector<IplImage*> &images, vector<IplImage*> &gaus);	

int cannytrainframe = 0;	
bool gauflag = 1;			
bool gaucannyflag = 1;


CvVideoWriter *writer;
int backgroundwidth;	//image width
int backgroundheight;	//image height

int imagepixelnumber;		
int forepixelnumber;		
double foreratio;			
int flashframenumber = 0;

RNG rnd = theRNG();
int rndNum;
int rndn[rndSize];
int rnd10[rndSize];
int rnd8[rndSize];	
int rdx = 0;		
int rdxf = 0;		

IplImage *cannybackground;	
IplImage *cannyforeground;	
IplImage *cannyframe;		

IplImage *backgroundsave;	

int photostart = 1000;	
int photonum;			

FILE *ParaFile = NULL; 
char * para_filename = "parameter.txt";

float frVal = 0.0;
float frNextXVal = 0.0;
float frNextYVal = 0.0;
float bgVal = 0.0;
float bgNextXVal = 0.0;
float bgNextYVal = 0.0;
float frGx = 0.0;
float frGy = 0.0;
float bgGx = 0.0;
float bgGy = 0.0;
float frMagnitude = 0.0;
float bgMagnitude = 0.0;
float diffMagnitude = 0.0;
float frMaxGradDir = 0.0;
float bgMaxGradDir = 0.0;
float angle = 0.0;


bool get_bit(unsigned int val, int position)
{
	return (val & (1 << position)) >> position;
}

vibook::vibook( int img_w, int img_h )
{
    /* default parameter */ 
	param.train_num		= 1;		//training frame number
	param.alpha			= 0.5;		//0.5 bright 0.4~0.7
	param.beta			= 1.1;		//1.1 bright 1.1~1.5
	param.epsilon1		= 15;		//training threshold
	param.epsilon2		= 10;		//detection threshold
	param.epsilon3		= 20;		//compare neighbor threshold
	param.epsilon4		= 5;		//repair threshold
	param.epsilon_tex	= 5;
	param.t_delete		= 1000;		//filter threshold
	
	frame_num = 0;			//frame number
	cannydenominator = 20;	
	cannynumerator = 12;	

	rndNum = 0;
	for(int i=0;i<rndSize;i++)
    {
        rndn[i]=rnd(RANDPROBABILITY);	
        rnd8[i]=rnd(8);		
		rnd10[i] = rnd(10);
    }
	
	for( int i = 0; i < (img_w+2) * (img_h+2); i++ ) 
	{
		bg_model.push_back( vector< viword >() );
		hist_image.push_back(vector< int >());
	}

	for( int j = 0; j < img_w * img_h; j++ )
	{
		canny_model.push_back( vector< canny >() );
	}
	
	lbp_mat = Mat::zeros(img_h, img_w, CV_8UC1);
	lbp_mat_new = Mat::zeros(img_h, img_w, CV_8UC1);

	cbResult = cvCreateImage( cvSize( img_w, img_h ), 8, 1 );	//result image
	MfgImg = cvCreateImage( cvSize( img_w, img_h ), 8, 1 );		//repair image
 }

vibook::~vibook()
{
	cvReleaseImage( &cbResult );
	cvReleaseImage( &MfgImg);
	bg_model.clear();
	canny_model.clear();
}

int main()
{
	ParaFile = fopen( para_filename,"r");
	cout << para_filename << endl;

	if (ParaFile == NULL)
	{
		printf("Loading parameter failed.\n");
	}

	double time;	

	fscanf( ParaFile, "%lf", &time);
	cout << "Will try " << time << " time " << endl;

	vector<IplImage*> _grayimgArray; 
	vector<IplImage*> _imgArray; 
	vector<IplImage*> _gauArray; 
	_grayimgArray.clear();
	_imgArray.clear();
	_gauArray.clear();

	loadImages_gary("C:\\testImages\\Perception\\WaterSurface",_grayimgArray);
	loadImages("C:\\testImages\\Perception\\WaterSurface",_imgArray,_gauArray);
	cout << "Load photo sucesses" << endl;

	int width = _imgArray.at(0)->width;
	int height = _imgArray.at(0)->height;
	backgroundsave = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,3);
	cout << "Frame width : " << width << endl;
	cout << "Frame height : " << height << endl;
	backgroundwidth = width + 2;
	backgroundheight = height + 2;
	imagepixelnumber = width * height;
	
	cannybackground = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
	cannyforeground = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
	cannyframe = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);

	char AviFileName[]="Result.avi";
	int AviForamt = -1;
	int FPS = 20;
	CvSize AviSize = cvSize(width,height);
	int AviColor = 1;
	//writer=cvCreateVideoWriter(AviFileName,AviForamt,FPS,AviSize,AviColor);

	for(int i = 0; i < time; i++)
	{
		int train_frame = 0;
		photonum = photostart;
		

		cout << "Construction Background Sample...";
		vibook vibook(width,height);
		cout << "done" << endl;

		double temp;
		fscanf( ParaFile, "%lf", &temp);
		vibook.param.epsilon1 = temp;
		cout << (i+1) << "th epsilon1 is " << vibook.param.epsilon1 << endl;
		fscanf( ParaFile, "%lf", &temp);
		vibook.param.epsilon2 = temp;
		cout << (i+1) << "th epsilon2 is " << vibook.param.epsilon2 << endl;
		fscanf( ParaFile, "%lf", &temp);
		vibook.param.epsilon3 = temp;
		cout << (i+1) << "th epsilon3 is " << vibook.param.epsilon3 << endl;
		fscanf( ParaFile, "%lf", &temp);
		vibook.param.epsilon4 = temp;
		cout << (i+1) << "th epsilon4 is " << vibook.param.epsilon4 << endl;
		fscanf( ParaFile, "%lf", &temp);
		vibook.param.epsilon_tex = temp;
		cout << (i+1) << "th epsilon_tex is " << vibook.param.epsilon_tex << endl;
		fscanf( ParaFile, "%lf", &temp);
		vibook.param.t_delete = temp;
		cout << (i+1) << "th filter delete is " << vibook.param.t_delete << endl;
		fscanf( ParaFile, "%lf", &temp);
		vibook.cannydenominator = temp;
		cout << (i+1) << "th cannydenominator is " << vibook.cannydenominator << endl;
		fscanf( ParaFile, "%lf", &temp);
		vibook.cannynumerator = temp;
		cout << (i+1) << "th cannynumerator is " << vibook.cannynumerator << endl;

		frVal = 0.0;
		frNextXVal = 0.0;
		frNextYVal = 0.0;
		bgVal = 0.0;
		bgNextXVal = 0.0;
		bgNextYVal = 0.0;
		frGx = 0.0;
		frGy = 0.0;
		bgGx = 0.0;
		bgGy = 0.0;
		frMagnitude = 0.0;
		bgMagnitude = 0.0;
		diffMagnitude = 0.0;
		frMaxGradDir = 0.0;
		bgMaxGradDir = 0.0;
		angle = 0.0;

		//training
		while(train_frame < vibook.param.train_num)
		{

			if(gauflag == 1)
			{
				if(gaucannyflag == 1)
				{
					vibook.initialbackground(_gauArray[train_frame],_gauArray[train_frame],_grayimgArray[train_frame]); 
				}
				else
				{
					vibook.initialbackground(_gauArray[train_frame],_imgArray[train_frame],_grayimgArray[train_frame]); 
				}
			}
			else
			{
				if(gaucannyflag == 1)
				{
					vibook.initialbackground(_imgArray[train_frame],_gauArray[train_frame],_grayimgArray[train_frame]); 
				}
				else
				{
					vibook.initialbackground(_imgArray[train_frame],_imgArray[train_frame],_grayimgArray[train_frame]); 
				}
			}
			train_frame++;
			photonum++;
		}

		//detection
		while(train_frame < _imgArray.size())
		{
			if(gauflag == 1)
			{
				if(gaucannyflag == 1)
				{
					vibook.detect(_gauArray[train_frame],_gauArray[train_frame],_grayimgArray[train_frame]);
				}
				else
				{
					vibook.detect(_gauArray[train_frame],_imgArray[train_frame],_grayimgArray[train_frame]);
				}
			}
			else
			{
				if(gaucannyflag == 1)
				{
					vibook.detect(_imgArray[train_frame],_gauArray[train_frame],_grayimgArray[train_frame]);
				}
				else
				{
					vibook.detect(_imgArray[train_frame],_imgArray[train_frame],_grayimgArray[train_frame]);
				}
			}
			photonum++;
			train_frame++;

			cvShowImage("cbresult.jpg", vibook.cbResult);
			cvWaitKey(10);
		}

 		vibook.erase_filter();
		vibook.~vibook();
	}

	fclose(ParaFile);
	system("pause");
}

void vibook::calclbpimage(IplImage *gray, IplImage *lbp_image){

	// get lbp_image_now
	Mat dst = Mat(gray);		// image after preprocessing
	Mat lbp;					// lbp image
	GaussianBlur(dst, dst, Size(7, 7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
	lbp::OLBP(dst, lbp); // use the extended operator
	lbp_image = new IplImage(lbp);

	/*int w = gray->width;
	int h = gray->height;

	int ipos;
	int ivalue;
	for (int x = 0; x < h; x++)
	{
		for (int y = 0; y < w; y++)
		{
			ipos = x * w + y;
			ivalue = 0;
			
			if (x > 0 && y > 0)
				ivalue += 128 * (gray->imageData[ipos] > gray->imageData[(x - 1) * w + (y - 1)] ? 1 : 0);
			if (x > 0)
				ivalue += 64 * (gray->imageData[ipos] > gray->imageData[(x - 1) * w + y] ? 1 : 0);
			if (x > 0 && y < w - 1)
				ivalue += 32 * (gray->imageData[ipos] > gray->imageData[(x - 1) * w + (y + 1)] ? 1 : 0);
			if (y > 0)
				ivalue += 16 * (gray->imageData[ipos] > gray->imageData[x * w + (y - 1)] ? 1 : 0);
			if (y < w - 1)
				ivalue += 8 * (gray->imageData[ipos] > gray->imageData[x * w + (y + 1)] ? 1 : 0);
			if (x < h - 1 && y > 0)
				ivalue += 4 * (gray->imageData[ipos] > gray->imageData[(x + 1) * w + (y - 1)] ? 1 : 0);
			if (x < h - 1)
				ivalue += 2 * (gray->imageData[ipos] > gray->imageData[(x + 1) * w + y] ? 1 : 0);
			if (x < h - 1 && y < w - 1)
				ivalue += 1 * (gray->imageData[ipos] > gray->imageData[(x + 1) * w + (y + 1)] ? 1 : 0);

			lbp_image->imageData[ipos] = ivalue;

		}
	}
*/
}

void vibook::calclbpimagemat(Mat &gray_mat, Mat& lbp_mat){

	GaussianBlur(gray_mat, gray_mat, Size(7, 7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
	lbp::OLBP(gray_mat, lbp_mat); // use the extended operator

}

void vibook::calchistimage(IplImage *gray, vector < vector< int > > &hist_image){

	int w = gray->width;
	int h = gray->height;

	for (int x = 0; x < h; x++)
	{
		for (int y = 0; y < w; y++)
		{
			int i = (x + 1) * (w + 2) + (y + 1);
			hist_image[i].clear();

			if ( x == 0 )
			{
				if ( y == 0 )
				{
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y + 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y + 1]);
				}
				else if ( y == w - 1 )
				{
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y - 1]);
				} 
				else
				{
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y + 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y + 1]);
				}
			}
			else if ( x == gray->width - 1 )
			{
				if (y == 0)
				{
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y + 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y + 1]);
				}
				else if (y == w - 1)
				{
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y - 1]);
				}
				else
				{
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y + 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y + 1]);
				}
			} 
			else
			{
				if (y == 0)
				{
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y + 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y + 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y + 1]);
				}
				else if (y == w - 1)
				{
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y - 1]);
				}
				else
				{
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x - 1)*gray->widthStep + y + 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[x*gray->widthStep + y + 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y - 1]);
					hist_image[i].push_back((unsigned char)gray->imageData[(x + 1)*gray->widthStep + y + 1]);
				}
			}
		}
	}

}

void vibook::updatelbp(IplImage *lbp_image, IplImage *lbp_background, int pos, float alpha){

	int pixel_1 = (unsigned char)lbp_image->imageData[pos];
	int pixel_2 = (unsigned char)lbp_background->imageData[pos];

	lbp_background->imageData[pos] = (1 - alpha) * pixel_2 + alpha * pixel_1;

}

void vibook::updatelbpmat(Mat &lbp_mat, Mat &lbp_background_mat, int i, int j, float alpha){

	int pixel_1 = lbp_mat.at<unsigned char>(i, j);
	int pixel_2 = lbp_background_mat.at<unsigned char>(i, j);

	lbp_background_mat.at<unsigned char>(i, j) = (1 - alpha) * pixel_2 + alpha * pixel_1;

}

float vibook::comparelbp(IplImage *lbp_image, IplImage *lbp_background, int pos){

	float dist = 0;
	unsigned char pixel_1 = (unsigned char)lbp_image->imageData[pos];
	unsigned char pixel_2 = (unsigned char)lbp_background->imageData[pos];

	for (int i = 8; i > 0; i--)
	{
		dist += get_bit(pixel_1, i) == get_bit(pixel_2, i) ? 0 : 1;
	}

	dist = dist / 8;

	return dist;
}

float vibook::comparelbpmat(Mat &lbp_mat, Mat &lbp_background_mat, int i, int j){

	float dist = 0;
	unsigned char pixel_1 = lbp_mat.at<unsigned char>(i, j);
	unsigned char pixel_2 = lbp_background_mat.at<unsigned char>(i, j);

	for (int i = 8; i > 0; i--)
	{
		dist += get_bit(pixel_1, i) == get_bit(pixel_2, i) ? 0 : 1;
	}

	dist = dist / 8;

	return dist;

}

float vibook::comparehist(vector< int > hist_test, viword &hist_background){

	// compare test and a single viword

	int csize = hist_test.size();
	int cmin[9];
	for (int i = 0; i < csize; i++)
	{
		if (hist_test[i] < hist_background.hist[i])
			cmin[i] = hist_test[i];
		else
			cmin[i] = hist_background.hist[i];
	}
	int sum_test = 0, sum_background = 0, sum_compare = 0;
	for (int i = 0; i < csize; i++)
	{
		sum_test += hist_test[i];
		sum_background += hist_background.hist[i];
		sum_compare += cmin[i];
	}

	int temp;

	if (sum_test == 0 && sum_background == 0)
		return 0;
	else if (sum_test == 0 || sum_background == 0)
		return 1;

	if (sum_test < sum_background)
		temp = sum_compare / sum_test;
	else
		temp = sum_compare / sum_background;

	return temp;

}

void vibook::initialbackground( IplImage *img ,IplImage *cannyimg, IplImage *gray)
{
	// initial lbp_image
	lbp_image = cvCloneImage(gray);
	calclbpimage(gray, lbp_image);
	

	calclbpimagemat(Mat(gray), lbp_mat);

	// initial hist_image
	cout << "Begin initial hist...";
	calchistimage(lbp_image, hist_image);

	//Initial Canny
	cannytrainframe = 0;
	cout << "Begin initial canny...";
	cvCvtColor(cannyimg,cannyframe,CV_BGR2GRAY);
	cvCanny(cannyframe,cannybackground,50,150,3);
	for(int x = 0; x < cannybackground->height; x++)
	{
		for(int y = 0; y < cannybackground->width; y++)
		{
			int i = x * cannybackground->widthStep + y;
			int cannypixel = x * cannybackground->width + y;
			
			CvScalar xt = cvScalar( (unsigned char)cannybackground->imageData[i + 0], 
									(unsigned char)cannybackground->imageData[i + 1], 
									(unsigned char)cannybackground->imageData[i + 2], 
									0 );
			
			if(xt.val[0] == 0)
			{
				canny canny;
				canny.cannyvalue = 0;
				canny_model[cannypixel].push_back( canny );
			}
			else
			{
				canny canny;
				canny.cannyvalue = 1;
				canny_model[cannypixel].push_back( canny );
			}
		}
	}
	cout << "done" << endl;

	cout << "Begin initial background...";
	//Initial Background
	for(int x = 0; x < img->height; x++)
	{
		for(int y = 0; y < img->width; y++)
		{
			int pixel = (((x+1) * ((img->width)+2)))+y+1;

			if((x!=0) && (x!=((img->height)-1)) && (y!=0) && (y!=((img->width)-1)))	//Pixel非邊緣
			{
				pushviword(bg_model, hist_image,img,gray,pixel,x,y);
				pushviword(bg_model, hist_image,img, gray, pixel, x - 1, y - 1);
				pushviword(bg_model, hist_image,img,gray,pixel,x-1,y);
				pushviword(bg_model, hist_image,img,gray,pixel,x-1,y+1);
				pushviword(bg_model, hist_image,img,gray,pixel,x,y+1);
				pushviword(bg_model, hist_image,img,gray,pixel,x+1,y+1);
				pushviword(bg_model, hist_image,img,gray,pixel,x+1,y);
				pushviword(bg_model, hist_image,img,gray,pixel,x+1,y-1);
				pushviword(bg_model, hist_image,img,gray,pixel,x,y-1);
			}
			else
			{
				if(x==0)
				{
					if((y!=0) && (y!=img->width-1))
					{
						pushviword(bg_model, hist_image,img,gray,pixel,x,y);
						pushviword(bg_model, hist_image,img,gray,pixel,x,y+1);
						pushviword(bg_model, hist_image,img,gray,pixel,x+1,y+1);
						pushviword(bg_model, hist_image,img,gray,pixel,x+1,y);
						pushviword(bg_model, hist_image,img,gray,pixel,x+1,y-1);
						pushviword(bg_model, hist_image,img,gray,pixel,x,y-1);
					}
					else
					{
						if(y==0)
						{
							pushviword(bg_model, hist_image,img,gray,pixel,x,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x,y+1);
							pushviword(bg_model, hist_image,img,gray,pixel,x+1,y+1);
							pushviword(bg_model, hist_image,img,gray,pixel,x+1,y);
						}
						else
						{
							pushviword(bg_model, hist_image,img,gray,pixel,x,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x+1,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x+1,y-1);
							pushviword(bg_model, hist_image,img,gray,pixel,x,y-1);
						}
					}
				}
				else
				{
					if(x==((img->height)-1))
					{
						if((y!=0) && (y!=img->width-1))
						{
							pushviword(bg_model, hist_image,img,gray,pixel,x,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x,y-1);
							pushviword(bg_model, hist_image,img,gray,pixel,x-1,y-1);
							pushviword(bg_model, hist_image,img,gray,pixel,x-1,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x-1,y+1);
							pushviword(bg_model, hist_image,img,gray,pixel,x,y+1);
						}
						else
						{
							if(y==0)
							{
								pushviword(bg_model, hist_image,img,gray,pixel,x,y);
								pushviword(bg_model, hist_image,img,gray,pixel,x-1,y);
								pushviword(bg_model, hist_image,img,gray,pixel,x-1,y+1);
								pushviword(bg_model, hist_image,img,gray,pixel,x,y+1);
							}
							else
							{
								pushviword(bg_model, hist_image,img,gray,pixel,x,y);
								pushviword(bg_model, hist_image,img,gray,pixel,x,y-1);
								pushviword(bg_model, hist_image,img,gray,pixel,x-1,y-1);
								pushviword(bg_model, hist_image,img,gray,pixel,x-1,y);
							}
						}
					}
					else
					{
						if(y==0)
						{
							pushviword(bg_model, hist_image,img,gray,pixel,x,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x-1,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x-1,y+1);
							pushviword(bg_model, hist_image,img,gray,pixel,x,y+1);
							pushviword(bg_model, hist_image,img,gray,pixel,x+1,y+1);
							pushviword(bg_model, hist_image,img,gray,pixel,x+1,y);
						}
						else
						{
							pushviword(bg_model, hist_image,img,gray,pixel,x,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x+1,y);
							pushviword(bg_model, hist_image,img,gray,pixel,x+1,y-1);
							pushviword(bg_model, hist_image,img,gray,pixel,x,y-1);
							pushviword(bg_model, hist_image,img,gray,pixel,x-1,y-1);
							pushviword(bg_model, hist_image,img,gray,pixel,x-1,y);
						}
					}
				}
			}
		}
	}
	cout << "done" << endl;
	cout << "Begin filter...";
	
	initial_filter();
}

void vibook::initial_filter()	
{
	for( int i = 0; i < bg_model.size(); i++ ) 
	{
		for( int j = 0; j < bg_model[i].size(); j++ )
		{
			for( vector< viword >::iterator iter = bg_model[i].begin()+(j+1); iter != bg_model[i].end(); )
			{
				if( (color_dis(bg_model[i][j].vm,iter->vm) < param.epsilon1) && initialbrightness(bg_model[i][j].aux.i_min,iter->aux.i_min) && initialbrightness(bg_model[i][j].aux.i_max,iter->aux.i_max)) 
				{	
					bg_model[i][j].vm = cvScalar( (bg_model[i][j].vm.val[0]+iter->vm.val[0])/2 ,
												  (bg_model[i][j].vm.val[1]+iter->vm.val[1])/2 ,
												  (bg_model[i][j].vm.val[2]+iter->vm.val[2])/2 , 0);

					bg_model[i][j].aux.i_min = min(bg_model[i][j].aux.i_min,iter->aux.i_min);
					bg_model[i][j].aux.i_max = max(bg_model[i][j].aux.i_max,iter->aux.i_max);
					vector<viword>::iterator temp;
					temp = bg_model[i].erase( iter );
					iter = temp;
				}
				else
					iter++;
			}
		}
	}
	cout << "done" << endl;
}

void vibook::erase_filter()
{
	for( int i = 0; i < bg_model.size(); i++ )
	{
		bg_model[i].clear();
	}
	for( int j = 0; j < canny_model.size(); j++ )
	{
		canny_model[j].clear();
	}
}

void vibook::detect( IplImage *img ,IplImage *cannyimg , IplImage *gray)
{	
	frame_num++;
	cannytrainframe++;
	cvCvtColor(cannyimg,cannyframe,CV_BGR2GRAY);
	cvCanny(cannyframe,cannyframe,50,150,3);

	if(cannytrainframe < cannydenominator-1)
	{
		cvAnd(cannybackground,cannyframe,cannybackground);
		
		for(int x = 0; x < cannybackground->height; x++)
		{
			for(int y = 0; y < cannybackground->width; y++)
			{
				int i = x * cannyframe->widthStep + y;
				int cannypixel = x * cannyframe->width + y;
				CvScalar xt = cvScalar( (unsigned char)cannyframe->imageData[i + 0], 
										(unsigned char)cannyframe->imageData[i + 1], 
										(unsigned char)cannyframe->imageData[i + 2], 
										0 );
				if(xt.val[0] == 0)
				{
					canny canny;
					canny.cannyvalue = 0;
					canny_model[cannypixel].push_back( canny );
				}
				else
				{
					canny canny;
					canny.cannyvalue = 1;
					canny_model[cannypixel].push_back( canny );
				}
			}
		}
	}
	else
	{
		for(int x = 0; x < cannybackground->height; x++)
		{
			for(int y = 0; y < cannybackground->width; y++)
			{
				int i = x * cannybackground->widthStep + y;
				int cannypixel = x * cannyframe->width + y;
				CvScalar xt = cvScalar( (unsigned char)cannyframe->imageData[i + 0], 
										(unsigned char)cannyframe->imageData[i + 1], 
										(unsigned char)cannyframe->imageData[i + 2], 
										0 );
				if(xt.val[0] == 0)
				{
					canny canny;
					canny.cannyvalue = 0;
					canny_model[cannypixel].push_back( canny );
				}
				else
				{
					canny canny;
					canny.cannyvalue = 1;
					canny_model[cannypixel].push_back( canny );
				}

				int sum = 0;
				vector<canny>::iterator iter;
				for( iter = canny_model[cannypixel].begin(); iter != canny_model[cannypixel].end(); iter++ )
				{
					sum = sum + iter->cannyvalue;	
				}

				if(sum < cannynumerator)	
				{
					cannybackground->imageData[i + 0] = 0;
				}
				else						
				{
					cannybackground->imageData[i + 0] = 255;
				}

				canny_model[cannypixel].erase( canny_model[cannypixel].begin() );
			
			}
		}
	}

	cvSub(cannyframe,cannybackground,cannyforeground);
	
#ifdef CANCELCANNY
	for(int m = 0; m < cannyforeground->height; m++)
	{
		for(int n = 0; n < cannyforeground->width; n++)
		{
			cannyforeground->imageData[m*cannyforeground->widthStep+n] = 0;
		}
	}
#endif


#ifdef RECORDBACKGROUND
	for(int i = 1; i < backgroundheight - 1; i++)
	{
		for(int j = 1; j < backgroundwidth - 1; j++)
		{
			int index = i*backgroundwidth + j;
			
			int bsum = 0;
			int gsum = 0;
			int rsum = 0;

			for(int k = 0; k < bg_model[index].size(); k++)
			{
				int B = bg_model[index][k].vm.val[0];
				int G = bg_model[index][k].vm.val[1];
				int R = bg_model[index][k].vm.val[2];
				bsum = bsum + B;
				gsum = gsum + G;
				rsum = rsum + R;
			}
			backgroundsave->imageData[(i-1)*backgroundsave->widthStep+3*(j-1)] = rsum/bg_model[index].size();
			backgroundsave->imageData[(i-1)*backgroundsave->widthStep+3*(j-1)+1] = gsum/bg_model[index].size();
			backgroundsave->imageData[(i-1)*backgroundsave->widthStep+3*(j-1)+2] = bsum/bg_model[index].size();
		}
	}
#endif

	// get lbp_image_now
	IplImage* lbp_image_now = cvCloneImage(gray);
	calclbpimage(gray, lbp_image_now);
	calchistimage(lbp_image_now, hist_image);

	calclbpimagemat(Mat(gray), lbp_mat_new);

	imshow("lbp_now.jpg", lbp_mat_new);
	imshow("lbp.jpg", lbp_mat);
	waitKey(10);

	for( int x = 0; x < img->height; x++ )
	{
		for( int y = 0; y < img->width; y++ )
		{
			int i = (x * img->widthStep) + (3 * y);	
			
			int canny_i = (x * cannyforeground->widthStep) + y;	

			int pixel = (((x+1) * ((img->width)+2)))+y+1;	
			int imagepixel = (x * img->width) + y;	

			CvScalar xt = cvScalar( (unsigned char)img->imageData[i+2], 
									(unsigned char)img->imageData[i+1], 
									(unsigned char)img->imageData[i+0], 0 );

			CvScalar cannyxt = cvScalar( (unsigned char)cannyforeground->imageData[canny_i+0],
										 (unsigned char)cannyforeground->imageData[canny_i+1], 
										 (unsigned char)cannyforeground->imageData[canny_i+2], 0 );

			double vec_i = sqrt( xt.val[0] * xt.val[0] + xt.val[1] * xt.val[1] + xt.val[2] * xt.val[2] );

			frVal = (float)gray->imageData[x*gray->widthStep + y];
			frNextXVal = (float)gray->imageData[x*gray->widthStep + (y+1)];
			frNextYVal = (float)gray->imageData[(x+1)*gray->widthStep + y];
			frGx = abs(frVal-frNextXVal);
			frGy = abs(frVal-frNextYVal);
			frMagnitude = sqrt(frGx * frGx + frGy * frGy);

			/*Step 2*/
			double delta = 0;	
			vector<viword>::iterator iter;	
			for( iter = bg_model[pixel].begin(); iter != bg_model[pixel].end(); iter++ ) 
			{
				float ths1, ths2, ths3, ths4;
				ths1 = color_dis(xt, iter->vm) / param.epsilon2;				// 1. return 0-n 0
				ths2 = brightness(vec_i, iter->aux.i_min, iter->aux.i_max);		// 2. return 0-1 0
				ths3 = 1 - comparehist(hist_image[pixel], *iter);				// 4. return 0-1 0
				ths4 = comparelbpmat(lbp_mat_new, lbp_mat, x, y);		// 3. return 0-1 0

				if( ths4 < 0.8 )	// judge background or not
				{
					break;
				}
			}

			/*step)*/

			if( iter == bg_model[pixel].end() ) 
			{
				// foreground
				cbResult->imageData[imagepixel] = 255;	
			}

			else 
			{
				// update lbp
				updatelbpmat(lbp_mat_new, lbp_mat, x, y, 0.05);

				// background
				{
					iter->vm        = cvScalar( ((xt.val[0] + (iter->aux.f * iter->vm.val[0]))/(iter->aux.f + 1)),
												((xt.val[1] + (iter->aux.f * iter->vm.val[1]))/(iter->aux.f + 1)),
												((xt.val[2] + (iter->aux.f * iter->vm.val[2]))/(iter->aux.f + 1)),
												0 );

					iter->aux.i_min		= min( vec_i, iter->aux.i_min );		
					iter->aux.i_max		= max( vec_i, iter->aux.i_max );		
					iter->aux.f			= iter->aux.f + 1;	
					iter->aux.lamda		= max( iter->aux.lamda, frame_num-iter->aux.q );		
					iter->aux.p			= iter->aux.p;
					iter->aux.q			= frame_num;

					// update hist
					iter->histsize = hist_image[pixel].size();
					for (int i = 0; i < hist_image[pixel].size(); i++){
						int temp = (iter->hist[i] + hist_image[pixel].at(i)) / 2;
						iter->hist[i] = temp;
					}

					cbResult->imageData[imagepixel] = 0;
				}

				if(cannyxt.val[0] == 0)	
				{
					if(rndn[rdx] == 0)	
					{
						switch(rnd8[rdx])
						{
						case 0:
							{
								if((x > 0)&&(y > 0))
								{
									updatefromneighbor(bg_model, img, cannyforeground, x, y, x - 1, y - 1);
								}
							}
							break;

						case 1:	
							{
								if(x > 0)
								{
									updatefromneighbor(bg_model, img, cannyforeground, x, y, x - 1, y);
								}
							}
							break;

						case 2:	
							{
								if((x > 0)&&(y < ((cannyforeground->width)-1)))
								{
									updatefromneighbor(bg_model, img, cannyforeground, x, y, x - 1, y + 1);
								}
							}
							break;

						case 3:	
							{
								if(y < ((cannyforeground->width)-1))
								{
									updatefromneighbor(bg_model, img, cannyforeground, x, y, x, y + 1);
								}
							}
							break;
						
						case 4:	
							{
								if((x < ((cannyforeground->height)-1))&&(y < ((cannyforeground->width)-1)))
								{
									updatefromneighbor(bg_model, img, cannyforeground, x, y, x + 1, y + 1);
								}
							}
							break;

						case 5:	
							{
								if(x < ((cannyforeground->height)-1))
								{
									updatefromneighbor(bg_model, img, cannyforeground, x, y, x + 1, y);
								}
							}
							break;

						case 6:	
							{
								if((x < ((cannyforeground->height)-1))&&(y > 0))
								{
									updatefromneighbor(bg_model, img, cannyforeground, x, y, x + 1, y - 1);
								}
							}
							break;

						case 7:	
							{
								if(y > 0)
								{
									updatefromneighbor(bg_model, img, cannyforeground, x, y, x, y - 1);
								}
							}
							break;
						}

						rdx++;
						
						switch(rnd8[rdx])
						{
						case 0:
							{
								if((x > 0)&&(y > 0))
								{
									updatetoneighbor(bg_model, img, cannyforeground, x, y, x - 1, y - 1);
								}
							}
							break;

						case 1:	
							{
								if(x > 0)
								{
									updatetoneighbor(bg_model, img, cannyforeground, x, y, x - 1, y);
								}
							}
							break;

						case 2:	
							{
								if((x > 0)&&(y < ((cannyforeground->width)-1)))
								{
									updatetoneighbor(bg_model, img, cannyforeground, x, y, x - 1, y + 1);
								}
							}
							break;

						case 3:	
							{
								if(y < ((cannyforeground->width)-1))
								{
									updatetoneighbor(bg_model, img, cannyforeground, x, y, x, y + 1);
								}
							}
							break;
						
						case 4:	
							{
								if((x < ((cannyforeground->height)-1))&&(y < ((cannyforeground->width)-1)))
								{
									updatetoneighbor(bg_model, img, cannyforeground, x, y, x + 1, y + 1);
								}
							}
							break;

						case 5:	
							{
								if(x < ((cannyforeground->height)-1))
								{
									updatetoneighbor(bg_model, img, cannyforeground, x, y, x + 1, y);
								}
							}
							break;

						case 6:	
							{
								if((x < ((cannyforeground->height)-1))&&(y > 0))
								{
									updatetoneighbor(bg_model, img, cannyforeground, x, y, x + 1, y - 1);
								}
							}
							break;

						case 7:	
							{
								if(y > 0)
								{
									updatetoneighbor(bg_model, img, cannyforeground, x, y, x, y - 1);
								}
							}
							break;
						}
					}
				}
			}
			rdx = ((rdx + 1)%rndSize);
		}
	}

	filter(img,gray);

	forepixelnumber = cvCountNonZero(cbResult);
	foreratio = (double)forepixelnumber/(double)imagepixelnumber;
	
	if(foreratio >= REINITIALRATIO)
	{
		cout << "foreratio" << foreratio << endl;
		cout << "flashnumber" << flashframenumber << endl;

		if(flashframenumber == REINITIALFRAMENUM)
		{
			cout << "reinitial" << endl;
			cout << "begin erase_filter" << endl;
			erase_filter();
			cout << "end erase_filter" << endl;
			initialbackground(img,cannyimg,gray);
		}
		else
		{
			flashframenumber++;
		}
	}
	else
	{
		flashframenumber = 0;
	}

#ifdef REPAIR
	fgenhacne(MfgImg, cbResult, img);
	char strrepairhole[1024];
	sprintf(strrepairhole,"%s/%06d.jpg",pathrepairhole,photonum);
	cvSaveImage(strrepairhole,MfgImg);
	cvShowImage("MResult",MfgImg);
	//cvWriteFrame(writer,cbResult);
#endif

}

#ifdef REPAIR
void vibook::fgenhacne(IplImage *MfgImg, IplImage *fgImg, IplImage *orgImg)
{
	IplImage *TfgImg;
	bool change = true;
	//count1用來計算有多少個鄰居是255, count2用colordist來計算有多少個鄰居color與該pixel相近
	int count1 = 0, count2 = 0;
	int globalcount = 0;
	
	cvSet( MfgImg, cvScalar( 0 ) );
	TfgImg = cvCloneImage(fgImg);

	while(change)
	{
		change = false;
		globalcount++;

		//採用內縮pixel的方式
		for(int y = 1 ; y < TfgImg->height -1 ; y++)
		{
			for(int x = 1 ; x < TfgImg->width - 1 ; x++)
			{
				int imagepixel;
				count1 = 0;
				count2 = 0;
				for(int k = 0 ; k < NeighborSize ; k++)
				{
					imagepixel = ((y + neighbory[k]) * TfgImg->widthStep) + (x + neighborx[k]);
					//cur_xt表示目前pixel的R G B 值,nei_xt表示某鄰居的R G B 值
					int i = (y * orgImg->widthStep) + (3 * x);	//目前pixel的資訊
					CvScalar cur_xt = cvScalar( (unsigned char)orgImg->imageData[i+2], 
											    (unsigned char)orgImg->imageData[i+1], 
												(unsigned char)orgImg->imageData[i+0], 0 );
					int j = ((y + neighbory[k]) * orgImg->widthStep) + (3 * (x + neighborx[k]));	//目前pixel的鄰居資訊
					CvScalar nei_xt = cvScalar( (unsigned char)orgImg->imageData[j+2], 
												(unsigned char)orgImg->imageData[j+1], 
												(unsigned char)orgImg->imageData[j+0], 0 );
					double delta = color_dis( cur_xt, nei_xt);	//計算colordistance
					if( delta <= param.epsilon4 )	//如果目前pixel與鄰居夠近,則count2++
					{
						count2++;
					}
					
					//if((unsigned char)TfgImg->imageData[imagepixel] == 255 && delta <= param.epsilon4)
					if((unsigned char)TfgImg->imageData[imagepixel] == 255)
						count1++;
				}

				imagepixel = (y * TfgImg->widthStep + x);
				if((unsigned char)TfgImg->imageData[imagepixel] == 255)
					MfgImg->imageData[imagepixel] = 255;
				else if((unsigned char)TfgImg->imageData[imagepixel] == 0)
				{
					if(count1 >= 5 || (count1 >= 4 && count2 == 7))
					//if((count1 >= 4 && count2 == 7))
					//if(count1 >= 5)
					{
						MfgImg->imageData[imagepixel] = 255;
						change = true;
					}
				}
			}
		}
		
		cvReleaseImage(&TfgImg);
		TfgImg = cvCloneImage(MfgImg);
	}

	//因為內縮一個pixel,所以將最外一圈原封不動貼回結果圖
	for(int y = 0 ; y < MfgImg->height ; y++)
	{
		int sx = 0, ex = MfgImg->width;
		int imagepixel = (y * MfgImg->widthStep + sx);
		MfgImg->imageData[imagepixel] = fgImg->imageData[imagepixel];
		imagepixel = (y * MfgImg->widthStep + ex);
		MfgImg->imageData[imagepixel] = fgImg->imageData[imagepixel];
	}
	for(int x = 0 ; x < MfgImg->width ; x++)
	{
		int sy = 0, ey = MfgImg->height - 1;
		int imagepixel = (sy * MfgImg->widthStep + x);
		MfgImg->imageData[imagepixel] = fgImg->imageData[imagepixel];
		imagepixel = (ey * MfgImg->widthStep + x);
		MfgImg->imageData[imagepixel] = fgImg->imageData[imagepixel];
	}
}
#endif

double vibook::color_dis( CvScalar &xt, CvScalar &vm )
{
	double xm_dot_vm_pow = pow( xt.val[0] * vm.val[0] + xt.val[1] * vm.val[1] + xt.val[2] * vm.val[2], 2 );
	double vm_pow = ( vm.val[0] * vm.val[0] + vm.val[1] * vm.val[1] + vm.val[2] * vm.val[2] );
	double xt_pow = ( xt.val[0] * xt.val[0] + xt.val[1] * xt.val[1] + xt.val[2] * xt.val[2] );
	double p_pow;
	
	if( xm_dot_vm_pow == 0 )
		p_pow = ( xm_dot_vm_pow + 0.001 ) / ( vm_pow + 0.001 );
	else
		p_pow = xm_dot_vm_pow / vm_pow;

	return sqrt( abs( xt_pow - p_pow ) );
}

bool vibook::initialbrightness( double &i_one, double &i_two )
{
	if( (i_one - i_two) < 70 )
		return true;
	else 
		return false;
}

float vibook::brightness( double i, double i_min, double i_max )
{
	double i_low = param.alpha * i_max;
	double i_up  = min( param.beta * i_max, i_min / param.alpha ); 

	if (i >= i_low && i <= i_up){
		return abs(abs(abs(i - i_low) / abs(i_up - i_low)) - 0.5);
	}

	return 1;
}

void vibook::filter(IplImage *img, IplImage *gray)
{
	/* filter out background model's viword */ 
	for( int i = 0; i < bg_model.size(); i++ ) 
	{
		for( vector< struct viword >::iterator iter = bg_model[i].begin(); iter != bg_model[i].end(); ) 
		{
			if( frame_num - iter->aux.q > param.t_delete ) 
			{
				vector<viword>::iterator temp;
				temp = bg_model[i].erase( iter );
				iter = temp;
			}
			else
			{
				iter++;
			}
		}

#ifdef PIXELBACKGROUND
		if((bg_model[i].size()==0) && !(((i/backgroundwidth)==0) || ((i/backgroundwidth)==(backgroundheight-1)) || ((i%backgroundwidth)==0) || ((i%backgroundwidth)==(backgroundwidth-1))))	//代表此pixel為空  要重建此pixel的background
		{
			cout << "<" << ((i/backgroundwidth)+1) << "," << ((i%backgroundwidth)+1) << ">" << endl;
			pixelbackground(img,gray,i);
		}
#endif
	}
}

#ifdef PIXELBACKGROUND
void vibook::pixelbackground( IplImage *img, IplImage *gray, int bg )	//偵對單一pixel建立新的background
{
	//先將i轉換成圖片pixel
	int x = ((bg/backgroundwidth)-1);
	int y = ((bg%backgroundwidth)-1);
	cout << "pixelbackground" << endl;
	cout << "<x,y> = " << "<" << x << "," << y << ">" << endl;
	cout << endl;

	//下面跟initial的程式碼一樣
	if((x!=0) && (x!=((img->height)-1)) && (y!=0) && (y!=((img->width)-1)))	//Pixel非邊緣
	{
		pushviword(bg_model, hist_image,img,gray,bg,x,y);
		pushviword(bg_model, hist_image,img,gray,bg,x-1,y-1);
		pushviword(bg_model, hist_image,img,gray,bg,x-1,y);
		pushviword(bg_model, hist_image,img,gray,bg,x-1,y+1);
		pushviword(bg_model, hist_image,img,gray,bg,x,y+1);
		pushviword(bg_model, hist_image,img,gray,bg,x+1,y+1);
		pushviword(bg_model, hist_image,img,gray,bg,x+1,y);
		pushviword(bg_model, hist_image,img,gray,bg,x+1,y-1);
		pushviword(bg_model, hist_image,img,gray,bg,x,y-1);
	}
	else
	{
		if(x==0)
		{
			if((y!=0) && (y!=img->width-1))	
			{
				pushviword(bg_model, hist_image,img,gray,bg,x,y);
				pushviword(bg_model, hist_image,img,gray,bg,x,y+1);
				pushviword(bg_model, hist_image,img,gray,bg,x+1,y+1);
				pushviword(bg_model, hist_image,img,gray,bg,x+1,y);
				pushviword(bg_model, hist_image,img,gray,bg,x+1,y-1);
				pushviword(bg_model, hist_image,img,gray,bg,x,y-1);
			}
			else
			{
				if(y==0)	
				{
					pushviword(bg_model, hist_image,img,gray,bg,x,y);
					pushviword(bg_model, hist_image,img,gray,bg,x,y+1);
					pushviword(bg_model, hist_image,img,gray,bg,x+1,y+1);
					pushviword(bg_model, hist_image,img,gray,bg,x+1,y);
				}
				else	
				{
					pushviword(bg_model, hist_image,img,gray,bg,x,y);
					pushviword(bg_model, hist_image,img,gray,bg,x+1,y);
					pushviword(bg_model, hist_image,img,gray,bg,x+1,y-1);
					pushviword(bg_model, hist_image,img,gray,bg,x,y-1);
				}
			}
		}
		else
		{
			if(x==((img->height)-1))
			{
				if((y!=0) && (y!=img->width-1))	
				{
					pushviword(bg_model, hist_image,img,gray,bg,x,y);
					pushviword(bg_model, hist_image,img,gray,bg,x,y-1);
					pushviword(bg_model, hist_image,img,gray,bg,x-1,y-1);
					pushviword(bg_model, hist_image,img,gray,bg,x-1,y);
					pushviword(bg_model, hist_image,img,gray,bg,x-1,y+1);
					pushviword(bg_model, hist_image,img,gray,bg,x,y+1);
				}
				else
				{
					if(y==0)
					{
						pushviword(bg_model, hist_image,img,gray,bg,x,y);
						pushviword(bg_model, hist_image,img,gray,bg,x-1,y);
						pushviword(bg_model, hist_image,img,gray,bg,x-1,y+1);
						pushviword(bg_model, hist_image,img,gray,bg,x,y+1);
					}
					else
					{
						pushviword(bg_model, hist_image,img,gray,bg,x,y);
						pushviword(bg_model, hist_image,img,gray,bg,x,y-1);
						pushviword(bg_model, hist_image,img,gray,bg,x-1,y-1);
						pushviword(bg_model, hist_image,img,gray,bg,x-1,y);
					}
				}
			}
			else
			{
				if(y==0)
				{
					pushviword(bg_model, hist_image,img,gray,bg,x,y);
					pushviword(bg_model, hist_image,img,gray,bg,x-1,y);
					pushviword(bg_model, hist_image,img,gray,bg,x-1,y+1);
					pushviword(bg_model, hist_image,img,gray,bg,x,y+1);
					pushviword(bg_model, hist_image,img,gray,bg,x+1,y+1);
					pushviword(bg_model, hist_image,img,gray,bg,x+1,y);
				}
				else
				{
					pushviword(bg_model, hist_image,img,gray,bg,x,y);
					pushviword(bg_model, hist_image,img,gray,bg,x+1,y);
					pushviword(bg_model, hist_image,img,gray,bg,x+1,y-1);
					pushviword(bg_model, hist_image,img,gray,bg,x,y-1);
					pushviword(bg_model, hist_image,img,gray,bg,x-1,y-1);
					pushviword(bg_model, hist_image,img,gray,bg,x-1,y);
				}
			}
		}
	}
}
#endif

void vibook::pushviword(vector < vector< viword > > &bg_model, vector < vector< int > > &hist_image, IplImage *img, IplImage *gray, int pixel, int x, int y)
{
	int i = x * img->widthStep + 3 * y;
	CvScalar xt = cvScalar( (unsigned char)img->imageData[i + 2], 
							(unsigned char)img->imageData[i + 1], 
							(unsigned char)img->imageData[i + 0], 
							0 );
	double vec_i = sqrt( xt.val[0] * xt.val[0] + xt.val[1] * xt.val[1] + xt.val[2] * xt.val[2] );

	frVal = (float)gray->imageData[x*gray->widthStep + y];
	frNextXVal = (float)gray->imageData[x*gray->widthStep + (y+1)];
	frNextYVal = (float)gray->imageData[(x+1)*gray->widthStep + y];
	frGx = abs(frVal-frNextXVal);
	frGy = abs(frVal-frNextYVal);
	frMagnitude = sqrt(frGx * frGx + frGy * frGy);

	viword viword;
	viword.vm			= cvScalar( xt.val[0], xt.val[1], xt.val[2],0 );
	viword.aux.i_min	= vec_i;
	viword.aux.i_max	= vec_i;
	viword.aux.f		= 1;
	viword.aux.lamda	= 0;
	viword.aux.p		= frame_num;
	viword.aux.q		= frame_num;
	
	viword.histsize = hist_image[pixel].size();
	for (int i = 0; i < hist_image[pixel].size(); i++)
		viword.hist[i] = hist_image[pixel].at(i);

	bg_model[pixel].push_back( viword );
}

void vibook::updatetoneighbor(vector < vector< viword > > &bg_model, IplImage *img, IplImage *cannyforeground, int x, int y, int nx, int ny)
{
	// read data
	int me_pixel = (((x + 1) * ((img->width) + 2))) + y + 1;
	int me_size = bg_model[me_pixel].size();
	int todata = getfastrandom(me_size);
	
	CvScalar xt = bg_model[me_pixel].at(todata).vm;
	double vec_i = sqrt(xt.val[0] * xt.val[0] + xt.val[1] * xt.val[1] + xt.val[2] * xt.val[2]);

	// write data
	int tem_pixel = (((nx + 1) * ((img->width) + 2))) + nx + 1;
	int tem_i = (nx * cannyforeground->widthStep) + nx;
							
	CvScalar cannyneightorxt = cvScalar( (unsigned char)cannyforeground->imageData[tem_i+0], 
										 (unsigned char)cannyforeground->imageData[tem_i+1], 
										 (unsigned char)cannyforeground->imageData[tem_i+2], 0 );
	if(cannyneightorxt.val[0] == 0)
	{
		double tem_delta = 0;
		vector<viword>::iterator iter;	
		for( iter = bg_model[tem_pixel].begin(); iter != bg_model[tem_pixel].end(); iter++ ) 
		{
			tem_delta = color_dis( xt, iter->vm);	
			if( tem_delta <= param.epsilon3 && brightness( vec_i, iter->aux.i_min, iter->aux.i_max ) )	
				break;
		}

		if( iter == bg_model[tem_pixel].end() )
		{
			viword viword;
			viword.vm			= cvScalar( xt.val[0], xt.val[1], xt.val[2],0 );
			viword.aux.i_min	= vec_i;
			viword.aux.i_max	= vec_i;
			viword.aux.f		= 1;
			viword.aux.lamda	= frame_num-1;
			viword.aux.p		= frame_num;
			viword.aux.q		= frame_num;
			bg_model[tem_pixel].push_back( viword );
		}				
		else
		{
			iter->vm        = cvScalar( ((xt.val[0] + (iter->aux.f * iter->vm.val[0]))/(iter->aux.f + 1)),
										((xt.val[1] + (iter->aux.f * iter->vm.val[1]))/(iter->aux.f + 1)),
										((xt.val[2] + (iter->aux.f * iter->vm.val[2]))/(iter->aux.f + 1)),
										 0 );
			iter->aux.i_min		= min( vec_i, iter->aux.i_min );		//Imin更新
			iter->aux.i_max		= max( vec_i, iter->aux.i_max );		//Imax更新
			iter->aux.f			= iter->aux.f + 1;	//frequency
			iter->aux.lamda		= max( iter->aux.lamda, frame_num-iter->aux.q );		//MNRL更新
			iter->aux.p			= iter->aux.p;
			iter->aux.q			= frame_num;
		}
	}
}

void vibook::updatefromneighbor(vector < vector< viword > > &bg_model, IplImage *img, IplImage *cannyforeground, int x, int y, int nx, int ny)
{
	// read data
	int n_pixel = (((nx + 1) * ((img->width) + 2))) + ny + 1;
	int n_size = bg_model[n_pixel].size();
	int todata = getfastrandom(n_size);

	CvScalar xt = bg_model[n_pixel].at(todata).vm;
	double vec_i = sqrt(xt.val[0] * xt.val[0] + xt.val[1] * xt.val[1] + xt.val[2] * xt.val[2]);

	// write data
	int tem_pixel = (((x + 1) * ((img->width) + 2))) + y + 1;
	int tem_i = (nx * cannyforeground->widthStep) + ny;
							
	CvScalar cannyneightorxt = cvScalar( (unsigned char)cannyforeground->imageData[tem_i+0], 
										 (unsigned char)cannyforeground->imageData[tem_i+1], 
										 (unsigned char)cannyforeground->imageData[tem_i+2], 0 );
	if(cannyneightorxt.val[0] == 0)
	{
		double tem_delta = 0;
		vector<viword>::iterator iter;
		for( iter = bg_model[tem_pixel].begin(); iter != bg_model[tem_pixel].end(); iter++ )
		{
			tem_delta = color_dis( xt, iter->vm);
			if( tem_delta <= param.epsilon3 && brightness( vec_i, iter->aux.i_min, iter->aux.i_max ) )
				break;
		}
		if( iter == bg_model[tem_pixel].end() )
		{
			viword viword;
			viword.vm			= cvScalar( xt.val[0], xt.val[1], xt.val[2],0 );
			viword.aux.i_min	= vec_i;
			viword.aux.i_max	= vec_i;
			viword.aux.f		= 1;
			viword.aux.lamda	= frame_num-1;
			viword.aux.p		= frame_num;
			viword.aux.q		= frame_num;
			bg_model[tem_pixel].push_back( viword );
		}
							
		else
		{
			iter->vm        = cvScalar( ((xt.val[0] + (iter->aux.f * iter->vm.val[0]))/(iter->aux.f + 1)),
										((xt.val[1] + (iter->aux.f * iter->vm.val[1]))/(iter->aux.f + 1)),
										((xt.val[2] + (iter->aux.f * iter->vm.val[2]))/(iter->aux.f + 1)),
										 0 );
			iter->aux.i_min		= min( vec_i, iter->aux.i_min );		//Imin更新
			iter->aux.i_max		= max( vec_i, iter->aux.i_max );		//Imax更新
			iter->aux.f			= iter->aux.f + 1;	//frequency
			iter->aux.lamda		= max( iter->aux.lamda, frame_num-iter->aux.q );		//MNRL更新
			iter->aux.p			= iter->aux.p;
			iter->aux.q			= frame_num;
		}
	}
}

int vibook::getfastrandom(int ssize){

	int rw3, rw2, rw1;
	int pos;

	if ((ssize / 100) >= 1)
	{
		do
		{
			rw3 = rnd10[rndNum % 7433];
			rndNum++;
			rw2 = rnd10[rndNum % 7433];
			rndNum++;
			rw1 = rnd10[rndNum % 7433];
			rndNum++;
			pos = rw1 + rw2 << 1 + rw3 << 2;
		} while (pos >= ssize);

	}
	else if ((ssize / 10) >= 1)
	{
		do
		{
			rw2 = rnd10[rndNum % 7433];
			rndNum++;
			rw1 = rnd10[rndNum % 7433];
			rndNum++;
			pos = rw1 + rw2 << 1;
		} while (pos >= ssize);
	}
	else
	{
		do
		{
			rw1 = rnd10[rndNum % 7433];
			rndNum++;
			pos = rw1;
		} while (pos >= ssize);
	}

	return pos;
}

void loadImages(string path, vector<IplImage*> &images, vector<IplImage*> &gaus)
{
	struct _finddata_t  c_file;
	long fh;

	string tempPath1 = path + "\\*.*";

	IplImage *temp;
	IplImage *gg;

	if((fh = _findfirst(tempPath1.c_str(), &c_file)) != -1)
	{
		while(_findnext(fh, &c_file) == 0)
		{ 	     
			if(strncmp(c_file.name, ".", 1) != 0 && strncmp(c_file.name, "..", 2) != 0) 
			{
				printf("%s\n", c_file.name);
				string file = path + "\\" + c_file.name;
				//images.push_back(cvLoadImage(const_cast<char*>(file.c_str()), 3));
				temp = cvLoadImage(const_cast<char*>(file.c_str()), 3);
				gg = cvCreateImage(cvSize(temp->width,temp->height),IPL_DEPTH_8U,3);
				
				cvSmooth(temp, gg, CV_GAUSSIAN, 3 , 3);

				images.push_back(temp);
				gaus.push_back(gg);
			}
		}
	}
	
}

void loadVideo(string path, vector<IplImage*> &images, vector<IplImage*> &gaus)
{
	char video_path[_MAX_PATH];
	strcpy(video_path, path.c_str());
	cout << video_path << endl;
	CvCapture *capture;
	capture = cvCaptureFromAVI(video_path);

	IplImage *video;
	IplImage *frame;
	IplImage *gg;

	video = cvQueryFrame(capture);
	//frame = cvCloneImage(video);
	//frame = cvCreateImage(cvGetSize(video),IPL_DEPTH_8U,3);
	gg = cvCreateImage(cvGetSize(video),IPL_DEPTH_8U,3);
	cvSmooth(video, gg, CV_GAUSSIAN, 3 , 3);

	cvSaveImage("temp.jpg",video);
	frame = cvLoadImage("temp.jpg");
	images.push_back(frame);
	gaus.push_back(gg);

	while(video = cvQueryFrame(capture))
	{
		cvSaveImage("temp.jpg",video);
		frame = cvLoadImage("temp.jpg");
		//frame = cvCloneImage(video);
		gg = cvCreateImage(cvGetSize(video),IPL_DEPTH_8U,3);
		cvSmooth(video, gg, CV_GAUSSIAN, 3 , 3);
		images.push_back(frame);
		gaus.push_back(gg);
	}
}

void loadImages_gary(string path, vector<IplImage*> &images)
{
	struct _finddata_t  c_file;
	long fh;

	string tempPath = path + "\\*.*";

	IplImage *temp;

	if((fh = _findfirst(tempPath.c_str(), &c_file)) != -1)
	{
		while(_findnext(fh, &c_file) == 0)
		{ 	     
			if(strncmp(c_file.name, ".", 1) != 0 && strncmp(c_file.name, "..", 2) != 0) 
			{
				//printf("%s\n", c_file.name);
				string file = path + "\\" + c_file.name;
				temp = cvLoadImage(const_cast<char*>(file.c_str()), 0);
				images.push_back(temp);
			}
		}
	}
}