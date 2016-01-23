//============================================================================
// Name        : LPD.cpp
// Author      : Tuan Hoang
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iomanip>      // std::setfill, std::setw

#include "src/def.hpp"
#include "src/Tracking.hpp"
#include "src/LPR.hpp"
#include "src/LetterClassifier.hpp"
#include "src/HelpFnc.hpp"
#include "src/timer.hpp"
#include "src/cvwin.hpp"

// Open CV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

// Intel TBB
#include "tbb/tbb.h"

std::string letter_classifier_param_file;
std::string licensePlate_detection_config_file;
std::string video_file;
int SKIP_FRAME = 0;
bool PRINT_TIME = false;
int WAIT_TIME = 0;
int START_FRAME = 0;

int DEBUG_LEVEL = std::numeric_limits<int>::max();

inline void Help()
{
	std::cout << "Usage: ./main " << std::endl;
	std::cout << "	-detection <license_plate_detection_config_file> " << std::endl;
	std::cout << "	-video <video_file> " << std::endl;
	std::cout << "	-skip <num_of_SKIP_FRAME> " << std::endl;
	std::cout << "	-time " << std::endl;
	std::cout << "	-debug <DEBUG_LEVEL> " << std::endl;
	std::cout << "	-wait <WAIT_TIME> " << std::endl;
	std::cout << "	-start <START_FRAME> " << std::endl;
}

inline void ArgumentParser(int argc, char** argv)
{
	for( int i = 1; i < argc; i++ )
	{
		if( strcmp(argv[i],"-detection") == 0 )
		{
			i++;
			licensePlate_detection_config_file = argv[i];
		}
		else if( strcmp(argv[i],"-video") == 0 )
		{
			i++;
			video_file = argv[i];
		}
		else if( strcmp(argv[i],"-skip") == 0 )
		{
			i++;
			SKIP_FRAME = std::atoi(argv[i]);
		}
		else if( strcmp(argv[i],"-time") == 0 )
		{
			PRINT_TIME = true;
		}
		else if( strcmp(argv[i],"-debug") == 0 )
		{
			i++;
			DEBUG_LEVEL = std::atoi(argv[i]);
		}
		else if( strcmp(argv[i],"-wait") == 0 )
		{
			i++;
			WAIT_TIME = std::atoi(argv[i]);
		}
		else if( strcmp(argv[i],"-start") == 0 )
		{
			i++;
			START_FRAME = std::atoi(argv[i]);
		}
	}
}

int main(int argc, char** argv) {
	if (argc < 2)
	{
		Help();
		return 1;
	}
//	cvwin org_frame("Original");

	timer licen_timer("Detect LP: 	");
	timer digit_timer("Get digits:	");
	timer class_timer("Classify:	");


	DMESG("Parsing argument ... ", 1);
	ArgumentParser(argc, argv);

	DMESG("Initializing LPR ... ", 1);
	LPR lpr(licensePlate_detection_config_file);

	DMESG("Opening video ... ", 1);
	cv::VideoCapture cap(video_file);
	if (!cap.isOpened())  		// if not success, exit program
	{
		std::cerr << "ERROR: Cannot open the video file" << std::endl;
		return false;
	}
	int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	std::cout << "Video size: " << GREEN_TEXT << width << " x " << height << NORMAL_TEXT << std::endl;
//	int fps = cap.get(CV_CAP_PROP_FPS);
	cap.set(CV_CAP_PROP_POS_FRAMES, START_FRAME);

	cv::Rect crop(0, height*3/10, width, height/2);
	cv::Mat frame, detect_area, result;

	int lp_Cnt = 3541;
	int digit_cnt = 21050;
	while (true)
	{
		for ( int s = 0; s <= SKIP_FRAME; s++)
		{
			if (!cap.read(frame)) //if not success, break loop
			{
				std::cerr << "ERROR: Cannot read a frame from video stream" << std::endl;
				return false;
			}
		}

		DMESG(YELLOW_TEXT << "----- NEW FRAME ----- " << NORMAL_TEXT, 1);
		detect_area = frame(crop);
//		org_frame.display_frame(detect_area);

		licen_timer.start();
		lpr.DetectLP(detect_area);
		licen_timer.stop();
		if (PRINT_TIME)
			licen_timer.printm();

		digit_timer.start();
		lpr.FindLicenseNumber(detect_area);
		digit_timer.stop();
		if (PRINT_TIME)
			digit_timer.printm();

		lpr.ShowLPs();
//
//		for ( size_t i = 0; i < lpr.lp_candidates.size(); i++ )
//		{
//			SaveImage(lpr.lp_candidates[i].lp_img, "images/License_plates/lp_pos", lp_Cnt++);
//			for ( size_t j = 0; j < lpr.lp_candidates[i].digit_boxes.size(); j++ )
//			{
//				cv::Mat digit = lpr.lp_candidates[i].lp_img(lpr.lp_candidates[i].digit_boxes[j]);
//				SaveImage(digit, "images/Digits/digits_pos", digit_cnt++);
//			}
//		}

		if (cv::waitKey(WAIT_TIME) == 27)
		{
			std::cout << RED_TEXT << "STOP!!! " << NORMAL_TEXT << std::endl;
			break;
		}
	}

	cap.release();
	licen_timer.aprintm();
	digit_timer.aprintm();
	return EXIT_SUCCESS;
}
