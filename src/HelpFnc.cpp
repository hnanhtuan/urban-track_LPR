/*
 * HelpFnc.cpp
 *
 *  Created on: Jan 4, 2016
 *      Author: anhxtuan
 */

#include "HelpFnc.hpp"

namespace help {

std::string Num2String(int num, int padding_width)
{
	std::stringstream ss;
	ss << std::setfill('0') << std::setw(padding_width) << num;
	return ss.str();
}

void GetBiggerOrSmallerArea(const cv::Mat &img, const cv::Rect &crop, double scale_x, double scale_y, cv::Mat &scale_img)
{
	cv::Rect scale_crop;
	GetBiggerOrSmallerArea(img, crop, scale_x, scale_y, scale_crop);
	scale_img = img(scale_crop);
}

void GetBiggerOrSmallerArea(const cv::Mat &img, const cv::Rect &crop, double scale_x, double scale_y, cv::Rect &scale_crop)
{
	assert(std::signbit(scale_x) == std::signbit(scale_y));
	if (scale_x > 0)
		GetBiggerArea(img, crop, scale_x, scale_y, scale_crop);
	else
		GetSmallerArea(img, crop, -scale_x, -scale_y, scale_crop);
}

void GetSmallerArea(const cv::Mat &img, const cv::Rect &crop, double shrink_x, double shrink_y, cv::Mat &shrink_img)
{

	cv::Rect shrink_crop;
	GetBiggerArea(img, crop, shrink_x, shrink_y, shrink_crop);
	shrink_img = img(shrink_crop);
}

void GetSmallerArea(const cv::Mat &img, const cv::Rect &crop, double shrink_x, double shrink_y, cv::Rect &shrink_crop)
{
	assert(shrink_x >= 0);
	assert(shrink_y >= 0);
	int x = crop.x + crop.width*(shrink_x/2);
	int y = crop.y + crop.height*(shrink_y/2);
	int w = crop.width*(1 - shrink_x);
	int h = crop.height*(1 - shrink_y);

	shrink_crop = cv::Rect(x, y, w, h);
}

bool IsRectsOverlap(const cv::Rect &first, const cv::Rect &second)
{
	int w = std::max(first.br().x, second.br().x) - std::min(first.x, second.x);
	int h = std::max(first.br().y, second.br().y) - std::min(first.y, second.y);
	return !((w > (first.width + second.width)) || (h > (first.height +  second.height)));
}

void GetBiggerArea(const cv::Mat &img, const cv::Rect &crop, double extend_x, double extend_y, cv::Mat &crop_img)
{
	cv::Rect extend_crop;
	GetBiggerArea(img, crop, extend_x, extend_y, extend_crop);
	crop_img = img(extend_crop);
}

void GetBiggerArea(const cv::Mat &img, const cv::Rect &crop, double extend_x, double extend_y, cv::Rect &extend_crop)
{
	assert(extend_x >= 0);
	assert(extend_y >= 0);
	int x = crop.x - crop.width*(extend_x/2);
	int y = crop.y - crop.height*(extend_y/2);
	int w = crop.width*(1 + extend_x);
	int h = crop.height*(1 + extend_y);

	x = std::max(0, x);
	y = std::max(0, y);
	w = std::min(w, img.cols - x - 1);
	h = std::min(h, img.rows - y - 1);
	extend_crop = cv::Rect(x, y, w, h);
}

void Write2Text(const std::string &text_file_name, const std::string &msg, bool append)
{
	std::ofstream pos_list_file;
	if (append)
		pos_list_file.open(text_file_name.c_str(), std::ios::out | std::ios::app);
	else
		pos_list_file.open(text_file_name.c_str(), std::ios::out);

	pos_list_file << msg << std::endl;

	pos_list_file.close();
}

void DrawText(const std::string &msg, const cv::Point &loc, cv::Mat &img)
{
	int baseline=0;
	double font_size = 0.8;
	double font_thickness = 1.5;
	cv::Size textSize = cv::getTextSize(msg, CV_FONT_HERSHEY_COMPLEX, font_size, font_thickness, &baseline);

	// center the text
	cv::Point textOrg;
	if ((loc.x + textSize.width) > img.cols)
		textOrg = cv::Point(img.cols - textSize.width, (loc.y + textSize.height));
	else
		textOrg = cv::Point(loc.x, loc.y + textSize.height);
	cv::putText( img, msg, textOrg,
		CV_FONT_HERSHEY_COMPLEX, font_size, cv::Scalar(100, 255, 0), font_thickness);
}

void Filter1D(std::vector<int> &vec_data, cv::Mat &kernel, cv::Mat &dst)
{
	cv::Mat data = cv::Mat(vec_data);
	Filter1D(data, kernel, dst);
}

void Filter1D(cv::Mat &data, cv::Mat &kernel, cv::Mat &dst)
{
	if (data.rows > 1)
		data = data.reshape(1, 1);

	if (kernel.rows > 1)
		kernel = kernel.reshape(1, 1);

	if (kernel.type() != CV_32F)
		kernel.convertTo(kernel, CV_32F);

	if (data.type() != CV_32F)
		data.convertTo(data, CV_32F);

	int kernel_size = kernel.cols;
	assert(kernel_size % 2 == 1);

	cv::Mat padding_data = cv::Mat::zeros(data.rows, data.cols + int(kernel_size/2)*2, data.type());
	data.copyTo(padding_data(cv::Rect(int(kernel_size/2), 0, data.cols, data.rows)));

	dst = cv::Mat(data.size(), data.type());
	for ( int c = 0; c < data.cols; c++ )
	{
		dst.at<float>(c) = cv::sum(padding_data.colRange(c, c + kernel_size)*kernel.t())[0];
	}
}

void PlotChart(std::vector< int > data, std::string win_name)
{
	cv::Mat chart;
	int hist_h = 400;
	cv::normalize(data, data, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat() );
	chart = cv::Mat::zeros(hist_h, data.size(), CV_8UC3);
	int bin_w = 1;
	for( size_t i = 1; i < data.size(); i++ )
	{
		cv::line( chart, cv::Point( bin_w*(i-1), hist_h - data[i-1]),
				cv::Point( bin_w*(i), hist_h - data[i]),
				cv::Scalar( 255, 0, 0), 1, 8, 0  );
	}
	cv::imshow(win_name, chart);
}

void PlotChart(std::vector< int > data, cv::Mat &chart)
{
	int hist_h = 400;
	cv::normalize(data, data, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat() );
	chart = cv::Mat::zeros(hist_h, data.size(), CV_8UC3);
	int bin_w = 1;
	for( size_t i = 1; i < data.size(); i++ )
	{
		cv::line( chart, cv::Point( bin_w*(i-1), hist_h - data[i-1]),
				cv::Point( bin_w*(i), hist_h - data[i]),
				cv::Scalar( 255, 0, 0), 1, 8, 0  );
	}
	cv::imshow("Chart", chart);
}

void PlotChart(const cv::Mat &data, std::string win_name)
{
	cv::Mat chart;
	PlotChart(data, chart, win_name);
}

void PlotChart(const cv::Mat &src, cv::Mat &chart, std::string win_name)
{
	cv::Mat data;
	int hist_h = 400;
	src.convertTo(data, CV_32F);
	cv::normalize(data, data, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat() );

	chart = cv::Mat::zeros(hist_h, data.cols, CV_8UC3);
	int bin_w = 1;
	for( int i = 1; i < data.cols; i++ )
	{
		cv::Point p1( bin_w*(i-1), hist_h - data.at<float>(i-1));
		cv::Point p2( bin_w*(i), hist_h - data.at<float>(i));
		cv::line( chart, p1, p2, cv::Scalar( 255, 0, 0), 1, 8, 0  );
	}
	cv::imshow(win_name, chart);
}

void Padding(const cv::Mat &src, int pad_size_x, int pad_size_y, cv::Mat &dst)
{
	dst = cv::Mat::zeros(src.rows + pad_size_y*2, src.cols + pad_size_x*2, src.type());
	src.copyTo(dst(cv::Rect(pad_size_x, pad_size_y, src.cols, src.rows)));
}

void Rotate(const cv::Mat& src, double angle_radian, cv::Mat& dst)
{
	int width  = std::abs(std::cos(angle_radian)*src.cols) + std::abs(std::sin(angle_radian)*src.rows);
	int height = std::abs(std::sin(angle_radian)*src.cols) + std::abs(std::cos(angle_radian)*src.rows);
	cv::Point2f pt(width/2.0, height/2.0);
	double angle = angle_radian*180/CV_PI;
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
	cv::Mat temp;
	Padding(src, (width - src.cols)/2, (height - src.rows)/2, temp);
	cv::warpAffine(temp, dst, r, cv::Size(width, height));
}

void ImAdjust(const cv::Mat &src, cv::Mat &dst, int tol)
{
	cv::Vec2i in  = cv::Vec2i(0, 255);
	cv::Vec2i out = cv::Vec2i(0, 255);
	dst = src.clone();

	tol = std::max(0, std::min(100, tol));

	if (tol > 0)
	{
		// Compute in and out limits

		// Histogram
		std::vector<int> hist(256, 0);
		for (int r = 0; r < src.rows; ++r) {
			for (int c = 0; c < src.cols; ++c) {
				hist[src.at<uchar>(r,c)]++;
			}
		}

		// Cumulative histogram
		std::vector<int> cum = hist;
		for (size_t i = 1; i < hist.size(); ++i) {
			cum[i] = cum[i - 1] + hist[i];
		}

		// Compute bounds
		int total = src.rows * src.cols;
		int low_bound = total * tol / 100;
		int upp_bound = total * (100-tol) / 100;
		in[0] = std::distance(cum.begin(), std::lower_bound(cum.begin(), cum.end(), low_bound));
		in[1] = std::distance(cum.begin(), std::lower_bound(cum.begin(), cum.end(), upp_bound));
	}

	// Stretching
	float scale = float(out[1] - out[0]) / float(in[1] - in[0]);
	for (int r = 0; r < dst.rows; ++r)
	{
		for (int c = 0; c < dst.cols; ++c)
		{
			int vs = std::max(src.at<uchar>(r, c) - in[0], 0);
			int vd = std::min(int(vs * scale + 0.5f) + out[0], out[1]);
			dst.at<uchar>(r, c) = cv::saturate_cast<uchar>(vd);
		}
	}
}

void ImAdjust(const cv::Mat &src, cv::Mat &dst, cv::Vec2i in)
{
	cv::Vec2i out = cv::Vec2i(0, 255);
	dst = src.clone();

	// Stretching
	float scale = float(out[1] - out[0]) / float(in[1] - in[0]);
	for (int r = 0; r < dst.rows; ++r)
	{
		for (int c = 0; c < dst.cols; ++c)
		{
			int vs = std::max(src.at<uchar>(r, c) - in[0], 0);
			int vd = std::min(int(vs * scale + 0.5f) + out[0], out[1]);
			dst.at<uchar>(r, c) = cv::saturate_cast<uchar>(vd);
		}
	}
}

void DoHist(const cv::Mat &src, bool wait)
{
	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
	if (src.channels() == 3)
	{
		/// Separate the image in 3 places ( B, G and R )
		std::vector< cv::Mat> bgr_planes;
		cv::split( src, bgr_planes );

		cv::Mat b_hist, g_hist, r_hist;

		/// Compute the histograms:
		cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
		cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
		cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

		/// Normalize the result to [ 0, histImage.rows ]
		cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
		cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
		cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

		/// Draw for each channel
		for( int i = 1; i < histSize; i++ )
		{
			cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
					cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
					cv::Scalar( 255, 0, 0), 2, 8, 0  );
			cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
					cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
					cv::Scalar( 0, 255, 0), 2, 8, 0  );
			cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
					cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
					cv::Scalar( 0, 0, 255), 2, 8, 0  );
		}
	}
	else if (src.channels() == 1)
	{
		cv::Mat hist;
		/// Compute the histograms:
		cv::calcHist( &src, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

		/// Normalize the result to [ 0, histImage.rows ]
		cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
		/// Draw for each channel
		for( int i = 1; i < histSize; i++ )
		{
			cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
					cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
					cv::Scalar( 255, 0, 0), 2, 8, 0  );
		}
	}

	/// Display
	cv::namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	cv::imshow("calcHist Demo", histImage );

	if (wait)
		cv::waitKey(0);
}

void CreateDirectory(const std::string &path)
{
	if (!IsExist(path))
		boost::filesystem::create_directory(path);
}

void CopyNonZero(cv::Mat &src, cv::Mat &dst)
{
	cv::MatIterator_<uchar> it, dst_it;
	int non_zero = cv::countNonZero(src);
	dst = cv::Mat(1, non_zero, src.type());
	dst_it = dst.begin<uchar>();
	for( it = src.begin<uchar>(); it != src.end<uchar>(); ++it)
	{
		if ((*it) != 0)
		{
			*dst_it = *it;
			dst_it++;
		}
	}
}

void GetTrainingImages(const std::string input_dir, std::vector< std::vector< std::string > > &file_names, std::vector< std::string > &classes)
{
	namespace fs = boost::filesystem;
	std::vector< std::string > ImageFileTypes {".jpg", ".png", ".pgm", ".JPG"};
	fs::path full_path( fs::initial_path<fs::path>() );
	full_path = fs::system_complete( fs::path( input_dir ) );

	if ( !fs::exists( full_path ) )
	{
		std::cout << "\nNot found: " << full_path.native() << std::endl;
		return;
	}

	if ( fs::is_directory( full_path ) )
	{
		std::cout << "\nIn directory: " << full_path.native() << "\n\n";
		fs::directory_iterator end_iter;
		for ( fs::directory_iterator dir_itr( full_path ); dir_itr != end_iter; ++dir_itr )
		{
			try
			{
				if ( fs::is_directory( dir_itr->status() ) )
				{
					std::string dir_name = dir_itr->path().filename().native();
					if (!dir_name.empty())
					{
						const char *cstr = dir_name.c_str();
						if (cstr[0] != '.')
						{
							classes.push_back(dir_name);
							std::vector< std::string > fileNames;
							fileNames.clear();
							for ( fs::directory_iterator file_itr( dir_itr->path() ); file_itr != end_iter; ++file_itr )
							{
								try {
									if (fs::is_regular_file(file_itr->status()))
									{
										std::string ext = file_itr->path().extension().native();
										bool exists = (std::find(ImageFileTypes.begin(), ImageFileTypes.end(), ext) != ImageFileTypes.end());
										if ( exists )
											fileNames.push_back(file_itr->path().native());
									}
								}
								catch ( const std::exception & ex )
								{
									std::cout << file_itr->path().filename() << " " << ex.what() << std::endl;
								}
							}
							std::sort(fileNames.begin(), fileNames.end());
							file_names.push_back(fileNames);
						}
					}
				}
			}
			catch ( const std::exception & ex )
			{
				std::cout << dir_itr->path().filename() << " " << ex.what() << std::endl;
			}
		}
	}
	return;
}

void SaveImage(const cv::Mat &img, const std::string file_name_format, int file_num)
{
	// Save image as pgm
	std::vector< int > compression_params;			//vector that stores the compression parameters of the image
	compression_params.push_back(CV_IMWRITE_PXM_BINARY);
	compression_params.push_back(9);

	if (file_num >= 0)
	{
		std::stringstream ss;
		ss << std::setfill('0') << std::setw(5) << file_num;
		std::string img_filename = file_name_format + "_" + ss.str() + ".pgm";
		cv::imwrite(img_filename, img, compression_params);
		std::cout << "Save: " << RED_TEXT << img_filename << NORMAL_TEXT << std::endl;
	}
	else
	{
		cv::imwrite(file_name_format, img, compression_params);
		std::cout << "Save: " << RED_TEXT <<  file_name_format << NORMAL_TEXT << std::endl;
	}

}

void DrawBoxes(const cv::Mat &src, const std::vector< cv::Rect > boxes, cv::Mat &dst)
{
	dst = src.clone();
	if (dst.channels() == 1)
		cv::cvtColor(dst, dst, CV_GRAY2RGB);
	for ( size_t i = 0; i < boxes.size(); i++ )
	{
		cv::rectangle( dst, boxes[i], cv::Scalar(0, 20*i, 255), 1, 8, 0);
	}
}

void DrawBoxesAndShow(cv::Mat &src, std::vector< cv::Rect > boxes, std::string win_name, bool wait)
{
	cv::Mat img = src.clone();
	if (img.channels() == 1)
		cv::cvtColor(img, img, CV_GRAY2RGB);
	for ( size_t i = 0; i < boxes.size(); i++ )
	{
		cv::rectangle( img, boxes[i], cv::Scalar(50*i, 80*i, 255), 1, 8, 0);
	}
	cv::imshow(win_name, img);
	if (wait)
		cv::waitKey(0);
}

void RemoveFiles(const std::string &filename)
{
	boost::filesystem::remove(filename);
}

bool IsExist(const std::string &filename)
{
	return boost::filesystem::exists( filename );
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string CurrentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	// Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
	// for more information about date/time format
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

	return buf;
}

void StoreDataset(const std::string &filename, const cv::Mat& m)
{
	std::ofstream ofs(filename.c_str());
	boost::archive::binary_oarchive binOutArc(ofs);

	size_t elem_size = m.elemSize();
	size_t elem_type = m.type();

	binOutArc & m.cols;
	binOutArc & m.rows;
	binOutArc & elem_size;
	binOutArc & elem_type;

	const size_t data_size = m.cols * m.rows * elem_size;
	binOutArc & boost::serialization::make_array(m.ptr(), data_size);

	std::cout << "Store: " << filename << std::endl;
}

void StoreDataset(const std::string &filename, const std::vector< std::string > &vec)
{
	std::ofstream ofs(filename.c_str());
	boost::archive::binary_oarchive binOutArc(ofs);

	size_t vec_size = vec.size();
	binOutArc & vec_size;
	for ( size_t i = 0; i < vec.size(); i++ )
	{
		binOutArc & vec[i];
	}
	std::cout << "Store: " << filename << std::endl;
}

/** Serialization support for cv::Mat */
void LoadDataset(const std::string &filename, cv::Mat& m)
{
	std::ifstream ifs(filename.c_str());
	boost::archive::binary_iarchive binInArc(ifs);

	int cols, rows;
	size_t elem_size, elem_type;

	binInArc & cols;
	binInArc & rows;
	binInArc & elem_size;
	binInArc & elem_type;

	m.create(rows, cols, elem_type);

	size_t data_size = m.cols * m.rows * elem_size;
	binInArc & boost::serialization::make_array(m.ptr(), data_size);

	std::cout << "Load: " << filename << std::endl;
}

void LoadDataset(const std::string &filename, std::vector< std::string > &vec)
{
	std::ifstream ifs(filename.c_str());
	boost::archive::binary_iarchive binInArc(ifs);

	size_t vec_size;

	binInArc & vec_size;
	std::string vec_item;
	for ( size_t i = 0; i < vec_size; i++ )
	{
		binInArc & vec_item;
		vec.push_back(vec_item);
	}
	std::cout << "Load: " << filename << std::endl;
}

void CopyFile(const std::string &src_file, const std::string &dst_file)
{
	std::ifstream  src(src_file, std::ios::binary);
	std::ofstream  dst(dst_file, std::ios::binary);

	dst << src.rdbuf();
	std::cout << "Copy: " << src_file << " to " << dst_file << std::endl;
}

void Rotate3D(const cv::Mat &img, cv::Mat &dst, float theta, float gamma, float beta) {
//	float beta = gamma;		/* Due to the order of our rotations gamma (Y-axis) gets mapped to beta (Z-axis) */
//	theta = theta - 90;		/* Turn camera downward */

	/* Convert to rads */
	theta = theta * CV_PI / 180.0;
	gamma = gamma * CV_PI / 180.0;
	beta = beta * CV_PI / 180.0;

	cv::Mat A1 = (cv::Mat_<float>(4,3) <<
		1, 0, -img.cols/2,
		0, 1, -img.rows/2,
		0, 0,    0,
		0, 0,    1);

	// Rotation cv::Matrices around the X,Y,Z axis
	cv::Mat RX = (cv::Mat_<float>(4, 4) <<
		1,          0,           0, 0,
		0, cos(theta), -sin(theta), 0,
		0, sin(theta),  cos(theta), 0,
		0,          0,           0, 1);

	/* Normally we need to adjust for gamma (left/right camera deviation) but
	 * since X rotation is applied first our original gamma (Y) is then mapped to the Z
	 * axis */

	cv::Mat RY = (cv::Mat_<float>(4, 4) <<
		cos(beta), 	0, -sin(beta), 0,
		0, 			1, 			0, 0,
		sin(beta), 	0,  cos(beta), 0,
		0, 			0, 			0, 1);

	/* Gamma (y rotation) gets mapped into Beta (z) after x rotation */
	cv::Mat RZ = (cv::Mat_<float>(4, 4) <<
		cos(gamma), -sin(gamma),   0, 0,
		sin(gamma),  cos(gamma),   0, 0,
		0,          0,           1, 0,
		0,          0,           0, 1);

	// Composed rotation cv::Matrix with (RX,RY,RZ)
	cv::Mat R = RX * RY * RZ;

	// Translation cv::Matrix on the Z axis change dist will change the height
	cv::Mat T = (cv::Mat_<float>(4, 4) <<
		1, 0, 0, X_TRANSLATION + (beta*180/CV_PI)*1.5,
		0, 1, 0, Y_TRANSLATION ,
		0, 0, 1, Z_TRANSLATION,
		0, 0, 0, 1);

	std::cout << "X_TRANSLATION: " << (beta*180/CV_PI) << std::endl;
	// Camera Intrisecs cv::Matrix 3D -> 2D
	cv::Mat A2 = (cv::Mat_<float>(3,4) <<
		FOCAL_IN_PX,	0,				img.cols/2,	0,
		0,				FOCAL_IN_PX,	img.rows/2,	0,
		0,				0,				1,				0);

	cv::Mat H = A2 * (T * (R * A1));
	H = H.inv();

	cv::warpPerspective(img, dst, H, img.size(), cv::INTER_CUBIC | cv::WARP_INVERSE_MAP);
	//cout << "theta: " << theta << endl;
	//cout << "gamma: " << gamma << endl;
}

template<typename T>
void Cv_mat_to_arma_mat(const cv::Mat_<T>& cv_mat_in, arma::Mat<T>& arma_mat_out)
{
	cv::Mat_<T> temp(cv_mat_in.t()); //todo any way to not create a temporary?
	//This compiles on both but is not as nice
	arma_mat_out = arma::Mat<T>(reinterpret_cast<T*>(temp.data),
								static_cast<arma::uword>(temp.cols),
								static_cast<arma::uword>(temp.rows),
								true,
								true);
};

template<typename T>
void Arma_mat_to_cv_mat(const arma::Mat<T>& arma_mat_in, cv::Mat_<T>& cv_mat_out)
{
	cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat_in.n_cols),
							  static_cast<int>(arma_mat_in.n_rows),
							  const_cast<T*>(arma_mat_in.memptr())),
				  cv_mat_out);
};
}
