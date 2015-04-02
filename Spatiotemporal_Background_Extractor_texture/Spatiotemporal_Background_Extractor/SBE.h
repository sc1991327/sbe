#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using std::vector;

using namespace std;
using namespace cv;

#define rndSize 7433
#define	NeighborSize	8

int neighborx[] = {-1, 0, +1,
				   -1, +1,
				   -1, 0, +1};
int neighbory[] = {-1, -1, -1,
					0, 0,
				   +1, +1, +1};

struct canny
{
	bool cannyvalue;
};

struct aux
{
	double i_max;
	double i_min;
	int f;	//frequency
	int lamda;	//MNLR
	int p;
	int q;
};

//viword
struct viword
{
	CvScalar	vm; /* rgb vector */
 	double		sigma;/* variance */
	int			hist[9];
	int			histsize;
	aux			aux;
};

struct vibook_param
{
	int train_num;	
	double alpha;	
	double beta;	
	int epsilon1;	
	int epsilon2;	
	int epsilon3;	
	int epsilon4;	
	int epsilon_tex;
	int t_delete;	
};

class vibook
{
public:
	vibook( int img_w, int img_h );
	~vibook();

	void initialbackground( IplImage *img ,IplImage *cannyimg, IplImage *gray );	
	void detect( IplImage *img ,IplImage *cannyimg, IplImage *gray);				
	void erase_filter();											

public:
	struct vibook_param param;		
	IplImage *cbResult;				
	IplImage *MfgImg;				
	int cannydenominator;			
	int cannynumerator;			

private:
	inline double color_dis( CvScalar &xt, CvScalar &vm );			
	inline bool initialbrightness( double &i_one, double &i_two );	
	inline float brightness(double i, double i_min, double i_max);
	inline void initial_filter();									
	inline bool textdis(int a, int b);

	inline void calclbpimage(IplImage *gray, IplImage *lbp_image);
	inline float comparelbp(IplImage *lbp_image, IplImage *lbp_background, int pos);
	inline void updatelbp(IplImage *lbp_image, IplImage *lbp_background, int pos, float alpha);

	inline void calclbpimagemat(Mat &gray_mat, Mat& lbp_mat);
	inline float comparelbpmat(Mat &lbp_mat, Mat &lbp_background_mat, int i, int j);
	inline void updatelbpmat(Mat &lbp_mat, Mat &lbp_background_mat, int i, int j, float alpha);

	inline void calchistimage(IplImage *gray, vector < vector< int > > &hist_image);
	inline float comparehist(vector< int > hist_test, viword &hist_background);

	inline void pushviword(vector < vector< viword > > &bg_model, vector < vector< int > > &hist_image, IplImage *img, IplImage *gray, int pixel, int x, int y);
	inline void updatetoneighbor(vector < vector< viword > > &bg_model,IplImage *img, IplImage *cannyforeground, int x, int y, int nx, int ny);
	inline void updatefromneighbor(vector < vector< viword > > &bg_model,IplImage *img, IplImage *cannyforeground, int x, int y, int nx, int ny);
	
	inline int getfastrandom(int ssize);

    void filter(IplImage *img, IplImage *gray);
	void pixelbackground( IplImage *img, IplImage *gray, int bg );	
	void fgenhacne(IplImage *MfgImg, IplImage *fgImg, IplImage *orgImg);

private:
	vector < vector< viword > > bg_model;	//BE
	vector < vector< canny > > canny_model;	//BGE
	vector < vector< int > > hist_image; // for each pixel's hist
	IplImage *lbp_image;
	Mat		lbp_mat;
	Mat		lbp_mat_new;

	int frame_num;

};