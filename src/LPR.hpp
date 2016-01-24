/*
 * LDR.hpp
 *
 *  Created on: Dec 25, 2015
 *      Author: anhxtuan
 */

#ifndef LDR_HPP_
#define LDR_HPP_

#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>

#include "def.hpp"
#include "LetterClassifier.hpp"
#include "HelpFnc.hpp"

// Intel TBB
#include "tbb/tbb.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define BOX_LP					( true )
#define LONG_LP					( false )

#define HEIGHT_WIDTH_RATIO		( 1.393 )


class LPR {
private:

	typedef struct LPR_Config {
		std::string box_cascade_classifier;
		std::string long_cascade_classifier;
		int lp_detect_area_height;
		std::vector< cv::Size > min_size, max_size;
		bool filter_by_size;
		bool tripple_check;

		int text_box_resize;
		double box_min_text_height, box_max_text_height;
		double long_min_text_height, long_max_text_height;
		double long_ideal_text_height;
		double min_text_ratio, max_text_ratio;
		double min_I_text_ratio, max_I_text_ratio;
		std::string letter_classifier_param_file;

		std::vector< cv::Scalar > LP_hsv_lower_range, LP_hsv_upper_range;
		double extend_area;
		double threshold;
		int num_rows, num_cols;

		void read(const std::string &filename)
		{
			cv::FileStorage fs(filename, cv::FileStorage::READ);
			fs["box_cascade_classifier"] >> box_cascade_classifier;
			fs["long_cascade_classifier"] >> long_cascade_classifier;
			filter_by_size = (fs["filter_by_size"] == "true");
			tripple_check = (fs["tripple_check"] == "true");

			cv::Size min_box, max_box;
			cv::Size min_long, max_long;

			fs["lp_detect_area_height"] >> lp_detect_area_height;
			fs["min_box"] >> min_box;
			fs["max_box"] >> max_box;
			fs["min_long"] >> min_long;
			fs["max_long"] >> max_long;

			min_size.push_back(min_box);
			min_size.push_back(min_long);

			max_size.push_back(max_box);
			max_size.push_back(max_long);

			text_box_resize = (int)fs["text_box_resize"];
			box_min_text_height = (double)fs["box_min_text_height"];
			box_max_text_height = (double)fs["box_max_text_height"];
			long_min_text_height = (double)fs["long_min_text_height"];
			long_max_text_height = (double)fs["long_max_text_height"];
			long_ideal_text_height = (double)fs["long_ideal_text_height"];
			min_text_ratio = (double)fs["min_text_ratio"];
			max_text_ratio = (double)fs["max_text_ratio"];
			min_I_text_ratio = (double)fs["min_I_text_ratio"];
			max_I_text_ratio = (double)fs["max_I_text_ratio"];
			fs["letter_classifier_param_file"] >> letter_classifier_param_file;

			cv::Scalar yellow_lower_range, red_lower_range, black_lower_range, white_lower_range;
			fs["yellow_lower_range"] >> yellow_lower_range;
			fs["red_lower_range"] >> red_lower_range;
			fs["black_lower_range"] >> black_lower_range;
			fs["white_lower_range"] >> white_lower_range;
			LP_hsv_lower_range.push_back(yellow_lower_range);
			LP_hsv_lower_range.push_back(red_lower_range);
			LP_hsv_lower_range.push_back(black_lower_range);
			LP_hsv_lower_range.push_back(white_lower_range);

			cv::Scalar yellow_upper_range, red_upper_range, black_upper_range, white_upper_range;
			fs["yellow_upper_range"] >> yellow_upper_range;
			fs["red_upper_range"] >> red_upper_range;
			fs["black_upper_range"] >> black_upper_range;
			fs["white_upper_range"] >> white_upper_range;
			LP_hsv_upper_range.push_back(yellow_upper_range);
			LP_hsv_upper_range.push_back(red_upper_range);
			LP_hsv_upper_range.push_back(black_upper_range);
			LP_hsv_upper_range.push_back(white_upper_range);

			extend_area = (double)fs["extend_area"];
			threshold = (double)fs["threshold"];
			num_cols = (int)fs["num_cols"];
			num_rows = (int)fs["num_rows"];

			fs.release();
		}

		void write(const std::string &filename)
		{
			cv::FileStorage fs(filename, cv::FileStorage::WRITE);
			fs << "box_cascade_classifier" << box_cascade_classifier;
			fs << "long_cascade_classifier" << long_cascade_classifier;

//			fs << "min_box" << min_box;
//			fs << "max_box" << max_box;
		}
	} LPR_Config;

	LetterClassifier classifier;

	typedef enum LP_Color {
		YELLOW,
		BLACK,
		RED,
		WHITE,
		UNKNOWN
	} LP_Color;

	typedef struct Digit_Candidate {
		Digit_Candidate(cv::Mat img, cv::Rect bbox, int row) :
			box(cv::Rect(bbox)),
			digit_img(img(bbox).clone()),
			str(""),
			adjust_str(""),
			row(row),
			idx(0),
			center(cv::Point(bbox.x + bbox.width/2, bbox.y + bbox.height/2)) {};

		cv::Rect box;
		cv::Mat digit_img;
		std::string str;
		std::string adjust_str;
		int row, idx;
		cv::Point center;

		void Update(cv::Mat img, cv::Rect bbox)
		{
			cv::Point new_center = cv::Point(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
			center = new_center;
			box = cv::Rect(bbox);
			digit_img = img(bbox).clone();
		}
	} Digit_Candidate;

	typedef struct LP_Candidate {
		LP_Candidate(cv::Mat img, cv::Rect box, int count) :
			id(count),
			box_long(box.width/box.height < 3),
			bounding_box(cv::Rect(box)),
			orig_lp_img(img.clone()),
			lp_img(img.clone()),
			color(UNKNOWN),
			digit_candidates(),
			license_number(""),
			matched(true),
			pos(cv::Point2f(float(box.x + box.width/2), float(box.y + box.height/2))){};

		int 							id;
		bool 							box_long;		// True: box; False: long
		cv::Rect 						bounding_box;
		cv::Mat 						orig_lp_img;
		cv::Mat 						lp_img;
		LP_Color 						color;
		std::vector< Digit_Candidate > 	digit_candidates;
		std::string 					license_number;

		bool 							matched;
		cv::Point2f						pos;

		void AddDigit(cv::Rect &bbox, int row)
		{
			Digit_Candidate digit(lp_img, bbox, row);
			digit_candidates.push_back(digit);
		}

		void Update(cv::Mat &img, cv::Rect &box)
		{
			bounding_box = cv::Rect(box);
			orig_lp_img = img.clone();
			lp_img = img.clone();
			pos = cv::Point2f((float)(box.x + box.width/2), (float)(box.y + box.height/2));
		}

		std::vector< cv::Rect > GetDigitBoxes()
		{
			std::vector< cv::Rect > digit_boxes;
			for ( size_t i = 0; i < digit_candidates.size(); i++ )
			{
				digit_boxes.push_back(digit_candidates[i].box);
			}
			return digit_boxes;
		}

	} LP_Candidate;

	int digit_cnt;

	int LP_count;

	LPR_Config config;

	cv::Ptr< cv::MSER > mserExtractor;

	std::vector< cv::CascadeClassifier > cascades;

	cv::CascadeClassifier digitsCascade;

	void Clustering(const cv::Mat &src, const cv::Mat &h_mask, cv::Mat &labels);

	void BestFitLP(const cv::Mat &src, int ncol, int nrow, double thres, cv::Mat &best_LP);

	void FilterOutlierByHeight(std::vector< cv::Rect > &boxes);

	void FindBoundContourBox(const cv::Mat &img, bool box_long, std::vector< cv::Rect > &boundRect);

	void FindLines(const bool box_long, std::vector< cv::Rect > &digit_boxes);

	float CalculateMatchDistance(const cv::Mat &img1, const cv::Mat &img2);

	void MatchingMethod(const cv::Mat &img, const LP_Candidate &candidate, cv::Rect &match, int match_method = CV_TM_SQDIFF);

	void CAMshiftMatching(const cv::Mat &img, const LP_Candidate &candidate, cv::Rect &match);

	void TextIsForeground(cv::Mat &img, std::vector< cv::Rect > &boxes);

	void LetterSpaceFilter(cv::Mat &data, int kernel_size, cv::Mat &kernel);

	static bool DigitCandidateTextLocCompare(Digit_Candidate first, Digit_Candidate second);
public:
	std::vector< LP_Candidate > lp_candidates;

	LPR(const std::string &config_filename);

	void MSERExtract(const cv::Mat &img, bool box_long, cv::Mat &dst);

	void DetectLP(const cv::Mat &img);

	void ShowLPs();

	void SaveLPs(const std::string &filename_template);

	std::string ClassifyDigits(LP_Candidate &candidate);

	void FindLicenseNumber(const cv::Mat &img);

	void FilterLPByColor(const cv::Mat &LP_img, cv::Mat &LP_crop_img);

	void GetDigitBoxes(cv::Mat &img, bool box_long,  std::vector< cv::Rect > &digit_boxes, cv::Mat &labels);
};
static bool BoxAreaCompare (cv::Rect first, cv::Rect second);

static bool BoxHeightCompare (cv::Rect first, cv::Rect second);

static bool BoxLocXCompare (cv::Rect first, cv::Rect second);

static bool LPTextLocCompare(cv::Rect first, cv::Rect second);

#endif /* LDR_HPP_ */
