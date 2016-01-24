/*
 * LDR.cpp
 *
 *  Created on: Dec 25, 2015
 *      Author: anhxtuan
 */

#include "LPR.hpp"

LPR::LPR(const std::string &config_filename)
{
	config.read(config_filename);
	std::cout << "Box  -- Min: " << config.min_size[0] << " - Max: " << config.max_size[0] << std::endl;
	std::cout << "Long -- Min: " << config.min_size[1] << " - Max: " << config.max_size[1] << std::endl;

//	std::cout << "Yellow -- Lower: " << config.LP_hsv_lower_range[0] << " - Upper: " << config.LP_hsv_upper_range[0] << std::endl;
//	std::cout << "Red    -- Lower: " << config.LP_hsv_lower_range[1] << " - Upper: " << config.LP_hsv_upper_range[1] << std::endl;
//	std::cout << "Black  -- Lower: " << config.LP_hsv_lower_range[2] << " - Upper: " << config.LP_hsv_upper_range[2] << std::endl;
//	std::cout << "White  -- Lower: " << config.LP_hsv_lower_range[3] << " - Upper: " << config.LP_hsv_upper_range[3] << std::endl;
	mserExtractor = cv::MSER::create();
	cv::CascadeClassifier cascade1, cascade2;
	if (!cascade1.load( config.box_cascade_classifier ))
		std::cerr << "Cannot load: " << config.box_cascade_classifier << std::endl;
	cascades.push_back(cascade1);
	if ( !cascade2.load( config.long_cascade_classifier ))
		std::cerr << "Cannot load: " << config.long_cascade_classifier << std::endl;
	cascades.push_back(cascade2);

	classifier.Initialize(config.letter_classifier_param_file);

	LP_count = 0;

	digit_cnt = 0;

	digitsCascade.load("digit_cascade.xml");
}

void LPR::MSERExtract(const cv::Mat &img, bool box_long,  cv::Mat &dst)
{
	std::vector<std::vector<cv::Point>> mserContours;
	std::vector<cv::Rect> boundRect;
	mserExtractor->detectRegions(img, mserContours, boundRect);

	img.copyTo(dst);
	if (dst.channels() == 1)
		cv::cvtColor(dst, dst, CV_GRAY2RGB);

	for (std::vector< cv::Rect >::iterator it = boundRect.begin(); it != boundRect.end();)
	{
		double ratio = (double)it->height/it->width;
		if ((box_long == BOX_LP) && ((it->height < (int)(config.text_box_resize * config.box_min_text_height))
				|| (it->height > (int)(config.text_box_resize * config.box_max_text_height))))
		{
			it = boundRect.erase(it);
		}
		else if ((box_long == LONG_LP) && ((it->height < (int)(config.text_box_resize * config.long_min_text_height))
				|| (it->height > (int)(config.text_box_resize * config.long_max_text_height))))
		{
			it = boundRect.erase(it);
		}
		else if ((ratio < config.min_text_ratio) || (ratio > config.max_text_ratio))
		{
			if ((ratio < config.min_I_text_ratio) || (ratio > config.max_I_text_ratio))
			{
				it = boundRect.erase(it);
			}
			else
			{
				it++;
			}
		}
		else
		{
			it++;
		}
	}
	help::DrawBoxesAndShow(dst, boundRect, "MSER");

//	for( size_t i = 0; i< mserContours.size(); i++ )
//	{
//		cv::drawContours(dst, mserContours, i, cv::Scalar(255, 0, 0), 4);
//	}

//	cv::imshow("Cropped", img);
//	cv::imshow("MSER", dst);
	cv::waitKey(0);
}

void LPR::DetectLP(const cv::Mat &src)
{
	cv::Mat img;
//	float ratio = float(src.rows)/config.lp_detect_area_height;
	float ratio = 1.;
	cv::resize(src, img, cv::Size(), ratio, ratio);
	DMESG("1/ Cascade detector", 1);
	std::vector< std::vector< cv::Rect > > LPboxes;
	LPboxes.resize(cascades.size());

	tbb::parallel_for(0, (int)cascades.size(), [&](int i) {
		if (config.filter_by_size)
		{
			cascades[i].detectMultiScale(img, LPboxes[i], 1.2, 3, CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH, config.min_size[i], config.max_size[i]);
		}
		else
			cascades[i].detectMultiScale(img, LPboxes[i], 1.2, 3, CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH);

		//Double check
		std::vector< cv::Rect >::iterator it;
		for ( it = LPboxes[i].begin(); it != LPboxes[i].end(); )
		{
			std::vector< cv::Rect > obj_double_check;
			cv::Rect crop(*it);
			crop.width = std::min(crop.width, img.cols - crop.x);
			crop.height = std::min(crop.height, img.rows - crop.y);
			cascades[i].detectMultiScale(img(crop), obj_double_check, 1.03, 3, CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH);

			// Redo the detection for the 3rd time
			if (obj_double_check.size() == 0)
			{
				if (config.tripple_check)
				{
					// Get bigger area to search for LP
					int x = it->x - it->width*0.1;
					x = std::max(0, x);
					int y = it->y - it->height*0.1;
					y = std::max(0, y);
					int w = it->width*1.2;
					w = std::min(w, img.cols - x);
					int h = it->height*1.2;
					h = std::min(h, img.rows - y);
					cv::Rect extend_crop(x, y, w, h);

					std::vector< cv::Rect > recheck_objs;
					cascades[i].detectMultiScale(img(extend_crop), recheck_objs, 1.03, 2, CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH);
					if (recheck_objs.size())
						it = LPboxes[i].erase(it);
					else
						it++;
				}
				else
					it++;
			}
			else if (obj_double_check.size() == 1)
			{
				it->x = it->x + obj_double_check[0].x;
				it->y = it->y + obj_double_check[0].y;
				it->width = obj_double_check[0].width;
				it->height = obj_double_check[0].height;
				it++;
			}
			else
				it++;
		}
	});

	DMESG("2/ Remove overlapping", 1);
	// Remove long LP if there is already a box LP.
	std::vector< cv::Rect >::iterator it;
	if (LPboxes[0].size() > 0)
	{
		for ( it = LPboxes[1].begin(); it != LPboxes[1].end(); )
		{
			bool overlap = false;

			for ( size_t i = 0; i < LPboxes[0].size(); i++ )
			{
				if (help::IsRectsOverlap((*it), LPboxes[0][i]))
				{
					overlap = true;
					break;
				}
			}

			if (overlap)
				it = LPboxes[1].erase(it);
			else
				it++;
		}
	}

	// Remove overlapping box_long
	for ( size_t i = 0; i < LPboxes.size(); i++ )
	{
		for (std::vector< cv::Rect >::iterator it1 = LPboxes[i].begin(); it1 != LPboxes[i].end(); it1++)
		{
			bool haveOverlap = false;
			for (std::vector< cv::Rect >::iterator it = it1 + 1; it != LPboxes[i].end(); )
			{
				if (help::IsRectsOverlap((*it), (*it1)))
				{
					it1->x = std::min(it1->x, it->x);
					it1->y = std::min(it1->y, it->y);
					it1->width  = std::max(it1->br().x, it->br().x) - it1->x;
					it1->height = std::max(it1->br().y, it->br().y) - it1->y;
					it = LPboxes[i].erase(it);
					haveOverlap = true;
				}
				else
					it++;
			}

			if (haveOverlap)
			{
				std::vector< cv::Rect > recheck_objs;
				cascades[i].detectMultiScale(img(*it1), recheck_objs, 1.03, 3, CV_HAAR_DO_CANNY_PRUNING | CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_ROUGH_SEARCH);
				if (recheck_objs.size() == 1)
					(*it1) = cv::Rect(recheck_objs[0]);
			}
		}
	}

	cv::Mat dst = img.clone();
	for ( size_t i = 0; i < LPboxes.size(); i++ )
	{
		for ( size_t j = 0; j < LPboxes[i].size(); j++ )
		{
			cv::rectangle( dst, LPboxes[i][j], cv::Scalar(0, 0, 255), 5, 8, 0);
		}
	}

	DMESG("3/ Mapping LP candidates between frames", 1);
	for ( size_t i = 0; i < lp_candidates.size(); i++)
	{
		lp_candidates[i].matched = false;
	}

	bool wait = false;
	for ( size_t i = 0; i < LPboxes.size(); i++ )
	{
		for ( size_t j = 0; j < LPboxes[i].size(); j++ )
		{
			bool newLP = true;
			cv::Rect LP_area_crop;
			help::GetBiggerArea(img, LPboxes[i][j], config.extend_area, config.extend_area, LP_area_crop);
			cv::Mat LP_area = img(LP_area_crop);

			for ( size_t k = 0; k < lp_candidates.size(); k++)
			{
				double matchDist = CalculateMatchDistance(lp_candidates[k].lp_img, LP_area);
				if (matchDist < 40.0)
				{
					lp_candidates[k].Update(LP_area, LPboxes[i][j]);
					lp_candidates[k].matched = true;
					newLP = false;
					DMESG("Match distance: " << GREEN_TEXT << matchDist << NORMAL_TEXT, 2);
					break;
				}
				else
				{
					DMESG("Candidate:  " << lp_candidates[k].bounding_box, 2);
					DMESG("New object: " << LPboxes[i][j], 2);
					DMESG("Match distance: " << YELLOW_TEXT << matchDist << NORMAL_TEXT, 2);
					wait = true;
				}
			}

			if (newLP)
			{
				DMESG("Add new candiate: " << LPboxes[i][j], 1);
				LP_Candidate lp(LP_area, LP_area_crop, LP_count++);
				lp_candidates.push_back(lp);
			}
		}
	}

	DMESG("4/ Find the LP cannot detect by cascade detector", 1);
	for ( size_t i = 0; i < lp_candidates.size(); i++)
	{
		if (!lp_candidates[i].matched)
		{
			cv::Rect temp;
			MatchingMethod(img, lp_candidates[i], temp);
//			CAMshiftMatching(img, lp_candidates[i], temp);
			cv::Mat lp_area = img(temp);
			lp_candidates[i].Update(lp_area, temp);
		}
		cv::rectangle( dst, lp_candidates[i].bounding_box, cv::Scalar(0, 255, 0), 2, 8, 0);
	}
	cv::imshow("Detect", dst);
//	if (wait)
//		cv::waitKey(0);
}

void LPR::MatchingMethod(const cv::Mat &img, const LP_Candidate &candidate, cv::Rect &match, int match_method)
{
	DMESG("Find Matching ... ", 1);
	cv::Mat templ = candidate.orig_lp_img;
	float text_box_ratio = float(candidate.digit_candidates[0].box.height)/templ.rows;
	if ((candidate.box_long && (text_box_ratio < 0.35)) || (!candidate.box_long && (text_box_ratio < 0.7)))
		cv::resize(templ, templ, cv::Size(), 0.96, 0.96);

	cv::Rect extend_crop;
	help::GetBiggerArea(img, candidate.bounding_box, 3, 3, extend_crop);
	cv::Mat sub_img = img(extend_crop);

	/// Source image to display
	cv::Mat img_display;
	sub_img.copyTo( img_display );

	/// Create the result matrix
	int result_cols = sub_img.cols - templ.cols + 1;
	int result_rows = sub_img.rows - templ.rows + 1;

	cv::Mat result = cv::Mat(result_cols, result_rows, CV_32FC1);

	if (templ.channels() == 3)
		cv::cvtColor(templ, templ, CV_RGB2GRAY);

	if (sub_img.channels() == 3)
		cv::cvtColor(sub_img, sub_img, CV_RGB2GRAY);
	/// Do the Matching and Normalize
	cv::matchTemplate( sub_img, templ, result, match_method );
	cv::normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal;
	cv::Point minLoc, maxLoc;
	cv::Point matchLoc;

	cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
		{ matchLoc = minLoc; }
	else
	    { matchLoc = maxLoc; }

	/// Show me what you got
	match = cv::Rect(matchLoc, cv::Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ));
	cv::rectangle( img_display, match, cv::Scalar::all(0), 2, 8, 0 );
	match.x += extend_crop.x;
	match.y += extend_crop.y;

//	cv::imshow( "Img", img_display );
}

void LPR::CAMshiftMatching(const cv::Mat &img, const LP_Candidate &candidate, cv::Rect &match)
{
	cv::Mat templ = candidate.orig_lp_img;

	cv::Rect extend_crop;
	help::GetBiggerArea(img, candidate.bounding_box, 3, 0.5, extend_crop);
	cv::Mat sub_img = img(extend_crop);

	std::vector< cv::Mat > channels;
	cv::Mat obj = candidate.lp_img.clone();
	cv::cvtColor(obj, obj, CV_RGB2HSV);
	cv::split(obj, channels);

	help::DoHist(channels[0]);
}

float LPR::CalculateMatchDistance(const cv::Mat &img1, const cv::Mat &img2)
{
	DMESG("Calculate Match Distance ... ", 1);
	cv::Mat im1, im2;
	double ratio = (double)200.0/img1.rows;

	cv::resize(img1, im1, cv::Size(), ratio, ratio);
	cv::resize(img2, im2, im1.size());

	if (im1.channels() == 3)
		cv::cvtColor(im1, im1, CV_RGB2GRAY);

	if (im2.channels() == 3)
		cv::cvtColor(im2, im2, CV_RGB2GRAY);

	cv::Ptr< cv::Feature2D > features = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_KAZE_UPRIGHT);;
	cv::Ptr< cv::DescriptorMatcher > descriptorMatcher = cv::DescriptorMatcher::create("BruteForce-L1");
	std::vector< cv::DMatch> matches;					// Match between img1 and img2
	std::vector< cv::KeyPoint> keyImg1, keyImg2;		// keypoint  for img1 and img2
	cv::Mat descImg1, descImg2;							// Descriptor for img1 and img2


	DMESG("		Detect and compute keypoints ... ", 1);
	features->detectAndCompute(im1, cv::Mat(), keyImg1, descImg1, false);
	features->detectAndCompute(im2, cv::Mat(), keyImg2, descImg2, false);

	double cumSumDist2=0;
	try
	{
		DMESG("		Match ... ", 1);
		descriptorMatcher->match(descImg1, descImg2, matches, cv::Mat());

		cv::Mat index;
		int nbMatch = int(matches.size());
		cv::Mat tab(nbMatch, 1, CV_32F);
		std::set<int> trainIdx;
		for (int i = 0; i < nbMatch; i++)
		{
			trainIdx.insert(matches[i].trainIdx);
			tab.at<float>(i, 0) = matches[i].distance;
		}
//		cv::imshow("im1", im1);
//		cv::imshow("im2", im2);
//		std::cout << "nbMatch: " << nbMatch <<std::endl;
//		std::cout << "trainIdx: " << trainIdx.size() <<std::endl;

		if (trainIdx.size() < nbMatch*0.4)
			return std::numeric_limits<double>::max();

		cv::sortIdx(tab, index, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);

		DMESG("		Calculate accumulative distance ... ", 1);
		for (int i = 0; i < std::min(20, nbMatch); i++ )
		{
//			cv::Point2d p=keyImg1[matches[index.at<int>(i, 0)].queryIdx].pt-keyImg2[matches[index.at<int>(i, 0)].trainIdx].pt;
			cumSumDist2 += matches[index.at<int>(i, 0)].distance;
		}
	}
	catch (cv::Exception& e)
	{
		cumSumDist2 = std::numeric_limits<double>::max();
	}
	return cumSumDist2;
}

void LPR::FindLicenseNumber(const cv::Mat &img)
{
	for ( std::vector< LP_Candidate >::iterator it = lp_candidates.begin(); it != lp_candidates.end(); )
	{
		std::vector< cv::Rect > digit_boxes;
		cv::Mat label;
		GetDigitBoxes(it->lp_img, it->box_long, digit_boxes, label);

		if (digit_boxes.size() <= 2)
		{
			it = lp_candidates.erase(it);
		}
		else
		{
			double ratio_sum = 0;
			for ( size_t i = 0; i < digit_boxes.size(); i++ )
			{
				ratio_sum += double(digit_boxes[i].height)/it->lp_img.rows;
			}
			double avg_ratio = ratio_sum/digit_boxes.size();
			DMESG(RED_TEXT << "avg text ratio: " << NORMAL_TEXT << avg_ratio, 3);
			if (!it->box_long)
			{
				DMESG(RED_TEXT << "Adjust lp area" << NORMAL_TEXT, 3);
				double scale = (avg_ratio/config.long_ideal_text_height) - 1.0;
				cv::Rect new_lp_crop;
				help::GetBiggerOrSmallerArea(img, it->bounding_box, scale*1.2, scale*1.2, new_lp_crop);
				cv::Mat new_lp_img = img(new_lp_crop);
				it->Update(new_lp_img, new_lp_crop);
				GetDigitBoxes(it->lp_img, it->box_long, digit_boxes, label);
			}

			TextIsForeground(label, digit_boxes);
			cv::imshow("Labels", label);
			it->digit_candidates.clear();


			for (std::vector< cv::Rect >::iterator box_it = digit_boxes.begin(); box_it != digit_boxes.end(); )
			{
				double ratio = double(box_it->height)/box_it->width;
				int num_letter = int(std::round(HEIGHT_WIDTH_RATIO/ratio*2));
//				DMESG("ratio :" << 1/ratio << " -- " << num_letter << " -- " << HEIGHT_WIDTH_RATIO/ratio*2, 3);
//				DMESG(BLUE_TEXT << "text: " << NORMAL_TEXT << double(box_it->height)/it->lp_img.rows, 3);

				if (num_letter == 1)
				{
					// Do nothing: it would be an I
				}
				else if (num_letter == 2)
				{

				}
				else if (num_letter == 3)
				{
					cv::Mat src = label(*box_it).clone();
					std::vector< int > data;
					int bin_w = 1;
					for ( int i = 0; i < src.cols - 2; i +=  bin_w)
					{
						int non_zero = cv::countNonZero(src(cv::Rect(i, 0, bin_w, src.rows)));
						data.push_back(non_zero);
					}

					int start = 10;
					int end = box_it->width - 10;
					int min = box_it->height;
					int min_idx;
					for ( int j = start; j < end; j++ )
					{
						if (data[j] < min)
						{
							min = data[j];
							min_idx = j;
						}
					}

					cv::Rect additional_box(box_it->x + min_idx, box_it->y, box_it->width - min_idx, box_it->height);
					digit_boxes.push_back(additional_box);
					box_it->width = min_idx;
					help::PlotChart(data, "Chart");
				}
				else if (num_letter == 4)
				{
					cv::Rect additional_box(box_it->x + box_it->width/2, box_it->y, box_it->width/2, box_it->height);
					box_it->width = box_it->width/2;
					digit_boxes.push_back(additional_box);
				}
				else if (num_letter == 6)
				{
					cv::Rect additional_box_1(box_it->x + box_it->width/3, box_it->y, box_it->width/3, box_it->height);
					digit_boxes.push_back(additional_box_1);
					cv::Rect additional_box_2(box_it->x + 2*box_it->width/3, box_it->y, box_it->width/3, box_it->height);
					digit_boxes.push_back(additional_box_2);
					box_it->width = box_it->width/3;
				}


				if (num_letter == 0)
					box_it = digit_boxes.erase(box_it);
				else
					box_it++;

			}


			std::sort(digit_boxes.begin(), digit_boxes.end(), LPTextLocCompare);

			// Find row number for boxes
			int row = 0;
			int row_0_cnt = 0;
			std::vector< cv::Point > centers;
			it->AddDigit(digit_boxes[0], row);
			centers.push_back(cv::Point(digit_boxes[0].x + digit_boxes[0].width/2, digit_boxes[0].y + digit_boxes[0].height/2));
			for ( size_t i = 1; i < digit_boxes.size(); i++ )
			{
				centers.push_back(cv::Point(digit_boxes[i].x + digit_boxes[i].width/2, digit_boxes[i].y + digit_boxes[i].height/2));
				if (it->box_long)
				{
					// Check if new line
					if (digit_boxes[i - 1].y + digit_boxes[i - 1].height/2 < digit_boxes[i].y)
						row++;
				}

				if (row == 0)
					row_0_cnt++;

				it->AddDigit(digit_boxes[i], row);
			}

			// Find missing digit boxes
			if (it->box_long)
			{
				for ( size_t i = 0; i < it->digit_candidates.size() - 1; i++ )
				{
					if (it->digit_candidates[i].row == it->digit_candidates[i + 1].row)
					{
						double ratio = double(it->digit_candidates[i].box.height)/(it->digit_candidates[i + 1].box.x - it->digit_candidates[i].box.br().x);
						int num_letter = int(std::round(HEIGHT_WIDTH_RATIO/ratio*2));
//						std::cout << BLUE_TEXT << "num_letter: " << NORMAL_TEXT << num_letter << " -- " << HEIGHT_WIDTH_RATIO/ratio*2 << std::endl;
						if ((HEIGHT_WIDTH_RATIO/ratio*2 > 0.8))
						{
							cv::Rect additional_box(it->digit_candidates[i].box.br().x, it->digit_candidates[i].box.y,
									(it->digit_candidates[i + 1].box.x - it->digit_candidates[i].box.br().x), it->digit_candidates[i].box.height);

							if ((num_letter == 1) || (num_letter == 2))
							{
								it->AddDigit(additional_box, 0);
							}
							else if ((num_letter == 3) || (num_letter == 5))
							{
								cv::Mat src = label(additional_box);
								std::vector< int > data;
								int bin_w = 1;
								for ( int i = 0; i < src.cols - 2; i +=  bin_w)
								{
									int non_zero = cv::countNonZero(src.colRange(i, i + bin_w));
									data.push_back(non_zero);
								}

								int start = 10;
								int end = additional_box.width - 10;
								int min = additional_box.height;
								int min_idx;
								for ( int j = start; j < end; j++ )
								{
									if (data[j] < min)
									{
										min = data[j];
										min_idx = j;
									}
								}

								cv::Rect additional_box_1(additional_box.x + min_idx, additional_box.y,
										additional_box.width - min_idx, additional_box.height);
								digit_boxes.push_back(additional_box_1);
								it->AddDigit(additional_box_1, 0);
								additional_box.width = min_idx;
								it->AddDigit(additional_box, 0);
								help::PlotChart(data, "Chart");
							}
							else if (num_letter == 4)
							{
								cv::Rect additional_box_1(additional_box.x + additional_box.width/2, additional_box.y,
										additional_box.width/2, additional_box.height);
								additional_box.width = additional_box.width/2;
								it->AddDigit(additional_box_1, 0);
								it->AddDigit(additional_box, 0);
							}
						}
					}
				}
			}
			else
			{
				for ( size_t i = 0; i < it->digit_candidates.size() - 1; i++ )
				{
					double ratio = double(it->digit_candidates[i].box.height)/(it->digit_candidates[i + 1].box.x - it->digit_candidates[i].box.br().x);
					int num_letter = int(std::round(HEIGHT_WIDTH_RATIO/ratio*2));
//					std::cout << BLUE_TEXT << "num_letter: " << NORMAL_TEXT << num_letter << " -- " << HEIGHT_WIDTH_RATIO/ratio*2 << std::endl;
					if ((HEIGHT_WIDTH_RATIO/ratio*2 > 0.8))
					{
						cv::Rect additional_box(it->digit_candidates[i].box.br().x, std::min(it->digit_candidates[i].box.y, it->digit_candidates[i + 1].box.y),
								(it->digit_candidates[i + 1].box.x - it->digit_candidates[i].box.br().x),
								std::max(it->digit_candidates[i].box.height, it->digit_candidates[i + 1].box.height));

						if ((num_letter == 1) || (num_letter == 2))
						{
							it->AddDigit(additional_box, 0);
						}
						else if (num_letter == 3)
						{
							cv::Mat src = label(additional_box);
							std::vector< int > data;
							int bin_w = 1;
							for ( int i = 0; i < src.cols - 2; i +=  bin_w)
							{
								int non_zero = cv::countNonZero(src.colRange(i, i + bin_w));
								data.push_back(non_zero);
							}

							int start = 10;
							int end = additional_box.width - 10;
							int min = additional_box.height;
							int min_idx;
							for ( int j = start; j < end; j++ )
							{
								if (data[j] < min)
								{
									min = data[j];
									min_idx = j;
								}
							}

							cv::Rect additional_box_1(additional_box.x + min_idx, additional_box.y,
									additional_box.width - min_idx, additional_box.height);
							digit_boxes.push_back(additional_box_1);
							it->AddDigit(additional_box_1, 0);
							additional_box.width = min_idx;
							it->AddDigit(additional_box, 0);
							help::PlotChart(data, "Chart");
						}
						else if (num_letter == 4)
						{
							cv::Rect additional_box_1(additional_box.x + additional_box.width/2, additional_box.y,
									additional_box.width/2, additional_box.height);
							additional_box.width = additional_box.width/2;
							it->AddDigit(additional_box_1, 0);
							it->AddDigit(additional_box, 0);
						}
					}
				}
			}

			int sum = 0;
			for ( size_t i = 0; i < it->digit_candidates.size(); i++ )
				sum += it->digit_candidates[i].box.height;
			int average_height = sum/int(it->digit_candidates.size());

			if (it->box_long)
			{
				cv::Vec4f first_line, second_line;
				bool first_line_valid = false, second_line_valid = false;
				std::vector< cv::Point > first_row(centers.begin(), centers.begin() + row_0_cnt);
				if (first_row.size() >= 2)
				{
					cv::fitLine(first_row, first_line, CV_DIST_L2, 0, 0.01, 0.01);
					first_line_valid = true;
//					std::cout << BLUE_TEXT << "First Line: " << NORMAL_TEXT << first_line[0] << "x + " << first_line[1] << "y + ";
					std::cout << -(first_line[0]*first_line[2] + first_line[1]*first_line[3]) << " = 0"<< std::endl;
				}
				std::vector< cv::Point > second_row(centers.begin() + row_0_cnt + 1, centers.end());
				if (second_row.size() >= 2)
				{
					cv::fitLine(second_row, second_line, CV_DIST_L2, 0, 0.01, 0.01);
					second_line_valid = true;
//					std::cout << BLUE_TEXT << "Second Line: " << NORMAL_TEXT  << second_line[0] << "x + " << second_line[1] << "y + ";
					std::cout << -(second_line[0]*second_line[2] + second_line[1]*second_line[3]) << " = 0"<< std::endl;
				}

				if (second_line_valid && (!first_line_valid))
				{
					first_line = cv::Vec4f(second_line);
				}
				else if ((!second_line_valid) && first_line_valid)
				{
					second_line = cv::Vec4f(first_line);
				}

				cv::Point pt1(0, first_line[3] - first_line[1]*first_line[2]/first_line[0]);
			}
			else
			{
				if ( centers.size() >= 3)
				{
					cv::Vec4f line;
					cv::fitLine(centers, line, CV_DIST_L2, 0, 0.01, 0.01);

					cv::Point pt1(0 /*line[2] - line[0]*line[2]/line[0]*/, line[3] - line[1]*line[2]/line[0]);
					cv::Point pt2(line[2] - line[0]*line[3]/line[1], 0 /*line[3] - line[1]*line[3]/line[1]*/);
//					std::cout << "Line: " << line[0] << "x + " << line[1] << "y + " << -(line[0]*line[2] + line[1]*line[3]) << " = 0"<< std::endl;
//					std::cout << "pt1: " <<  pt1 << std::endl;
//					std::cout << "pt2: " << pt2 << std::endl;

//					cv::Point pt3(line[2] - line[0]*1000, line[3] - line[1]*1000 - average_height/2);
//					cv::Point pt4(line[2] + line[0]*1000, line[3] + line[1]*1000 - average_height/2);
//					cv::Point pt5(line[2] - line[0]*1000, line[3] - line[1]*1000 + average_height/2);
//					cv::Point pt6(line[2] + line[0]*1000, line[3] + line[1]*1000 + average_height/2);

					cv::Mat disp_label = label.clone();
//					if (disp_label.channels() == 1)
//						cv::cvtColor(disp_label, disp_label, CV_GRAY2RGB);
//					cv::line(disp_label, pt1, pt2, cv::Scalar(255, 255, 0), 3);
//					cv::line(disp_label, pt3, pt4, cv::Scalar(255, 0, 0), 3);
//					cv::line(disp_label, pt5, pt6, cv::Scalar(255, 0, 0), 3);
//					cv::circle(disp_label, cv::Point(int(line[2]), int(line[3])), 3, cv::Scalar(0, 255, 0), -1);
//					cv::imshow("Line", disp_label);
					double angle = std::atan2(line[1], line[0]);
					help::Rotate(disp_label, angle, disp_label);
//					cv::imshow("Rotated Line", disp_label);

					cv::Rect crop_row(0, std::cos(angle)*(line[3] - line[1]*line[2]/line[0] - average_height/2), disp_label.cols, std::cos(angle)*average_height);
					cv::Mat rotated_row = disp_label(crop_row);
//					cv::imshow("Rotated Row", rotated_row);


				}
			}

//			std::sort(it->digit_candidates.begin(), it->digit_candidates.end(), DigitCandidateTextLocCompare);
			help::DrawBoxesAndShow(it->lp_img, it->GetDigitBoxes(), "Digits");
//			cv::waitKey(0);
//			std::cout << "Image size: " << it->lp_img.rows << " x " << it->lp_img.cols << std::endl;

			it->license_number = ClassifyDigits((*it));
			DMESG("ID: " << YELLOW_TEXT << it->id << NORMAL_TEXT <<  " -- License number: " << GREEN_TEXT << it->license_number << NORMAL_TEXT, 3);
//			cv::waitKey(0);
			it++;
		}
	}
}

void LPR::GetDigitBoxes(cv::Mat &src, bool box_long, std::vector< cv::Rect > &digit_boxes, cv::Mat &labels)
{
	cv::Mat img, sharpen_img, blur_img, thres_img, src_hsv;
	double ratio = (double)config.text_box_resize/src.rows;
	cv::resize(src, src, cv::Size(), ratio, ratio);
	cv::cvtColor(src, src_hsv, CV_RGB2HSV);
	cv::GaussianBlur(src, blur_img, cv::Size(0, 0), 5);
	cv::GaussianBlur(blur_img, blur_img, cv::Size(0, 0), 5);
	cv::addWeighted(src, 8, blur_img, -7, 0, sharpen_img);

//	cv::imshow("Sharpen", sharpen_img);

//	cv::Mat dst;
//	MSERExtract(img, box_long, dst);
	cv::cvtColor(sharpen_img, img, CV_RGB2HSV);

	std::vector< cv::Mat> hsv_planes;
	cv::split( img, hsv_planes );

	std::vector< cv::Mat> src_hsv_planes;
	cv::split( src_hsv, src_hsv_planes );

	int K = 2;
	cv::Mat centers, mask;
	cv::inRange(src_hsv_planes[0], cv::Scalar(115), cv::Scalar(130), mask);
	cv::bitwise_not(mask, mask);
//	cv::imshow("mask", mask);
	cv::Mat v_plane;
	hsv_planes[2].copyTo(v_plane, mask);

//	cv::imshow("v_plane", v_plane);
	if (v_plane.type() != CV_32F)
		v_plane.convertTo(v_plane, CV_32F);

	v_plane = v_plane.reshape(1, hsv_planes[2].rows*hsv_planes[2].cols);
	cv::kmeans(v_plane, K, labels, cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
			3, cv::KMEANS_PP_CENTERS, centers);

	labels = labels.reshape(1, hsv_planes[2].rows);
	labels.convertTo(labels, CV_8U, 255);

	cv::medianBlur(labels, labels, 5);

//	cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
//	cv::morphologyEx(labels, labels, cv::MORPH_CLOSE, kernel);
	FindBoundContourBox(labels, box_long, digit_boxes);
	FilterOutlierByHeight(digit_boxes);

//	cv::Mat draw_digit_boxes;
//	DrawBoxes(src, digit_boxes, draw_digit_boxes);
//	cv::imshow("Get digit boxes", draw_digit_boxes);
//	cv::waitKey(0);
}

void LPR::FindBoundContourBox(const cv::Mat &img, bool box_long, std::vector< cv::Rect > &boundRect)
{
	std::vector< std::vector< cv::Point > > contours_poly;
	std::vector< std::vector< cv::Point > > contours;
	std::vector< cv::Vec4i > hierarchy;

	cv::Mat tmp = img.clone();
	/// Find contours
	cv::findContours(tmp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	contours_poly.resize(contours.size());
	boundRect.resize(contours.size());

	for (unsigned int i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
	}

	for (std::vector< cv::Rect >::iterator it = boundRect.begin(); it != boundRect.end();)
	{
		double ratio = (double)it->height/it->width;
		if ((box_long == BOX_LP) && ((it->height < (int)(config.text_box_resize * config.box_min_text_height))
				|| (it->height > (int)(config.text_box_resize * config.box_max_text_height))))
		{
			it = boundRect.erase(it);
		}
		else if ((box_long == LONG_LP) && ((it->height < (int)(config.text_box_resize * config.long_min_text_height))
				|| (it->height > (int)(config.text_box_resize * config.long_max_text_height))))
		{
			it = boundRect.erase(it);
		}
		else if (ratio > config.max_text_ratio)
		{
			if  ((ratio < config.min_I_text_ratio) || (ratio > config.max_I_text_ratio))
			{
//				std::cout << "ratio: " << ratio << std::endl;
				it = boundRect.erase(it);
			}
			else
			{
				it++;
			}
		}
		else if (ratio < config.min_text_ratio)
		{
			it = boundRect.erase(it);
		}
		else
		{
			it++;
		}
	}
}

void LPR::TextIsForeground(cv::Mat &img, std::vector< cv::Rect > &boxes)
{
	int bg_value = -1;
	int non_zero_cnt = 0, zero_cnt = 0;
	for (size_t i = 0; i < boxes.size(); i++ )
	{
		cv::Rect box = boxes[i];
		if ((box.x > 0) && (box.y > 0) && (box.br().x < img.cols - 1) && (box.br().y < img.rows - 1))
		{
			for ( int r = box.y + 1; r < box.br().y; r++ )
			{
				non_zero_cnt += int(img.at<uchar>(r, box.x) > 0);
				non_zero_cnt += int(img.at<uchar>(r, box.br().x) > 0);
			}

			cv::Mat border = img(cv::Rect(box.x - 1, box.y - 1, box.width + 2, box.height + 2)).clone();
			cv::Mat zero = cv::Mat::zeros(box.size(), img.type());
			zero.copyTo(border(cv::Rect(1, 1, box.width, box.height)));
			non_zero_cnt = cv::countNonZero(border);
			zero_cnt = border.size().area() - box.area() - non_zero_cnt;
			bg_value = (non_zero_cnt > zero_cnt) ? 255 : 0;
			break;
		}
	}
	if (bg_value == 255)
		cv::bitwise_not(img, img);
}

void LPR::ShowLPs()
{
	if (lp_candidates.size() == 0)
	{
//		cv::destroyWindow("License Plate");
		return;
	}

	int num_box = 0, num_long = 0;
	int max_box_width = 0, max_long_width = 0;
	for ( size_t i = 0; i < lp_candidates.size(); i++)
	{
		if (lp_candidates[i].box_long)
		{
			num_box++;
			if (lp_candidates[i].lp_img.cols > max_box_width)
				max_box_width = lp_candidates[i].lp_img.cols;
		}
		else
		{
			num_long++;
			if (lp_candidates[i].lp_img.cols > max_long_width)
				max_long_width = lp_candidates[i].lp_img.cols;
		}
	}

	cv::Mat display = cv::Mat::zeros(config.text_box_resize*std::max(num_box, num_long), max_box_width + max_long_width, lp_candidates[0].lp_img.type());
	int box_cnt = 0, long_cnt = 0;
	for ( size_t i = 0; i < lp_candidates.size(); i++)
	{
		cv::Mat lp = lp_candidates[i].lp_img.clone();
		help::DrawText(help::Num2String(lp_candidates[i].id, 0), cv::Point(0, 0), lp);
		std::vector< cv::Rect > digit_boxes = lp_candidates[i].GetDigitBoxes();
		for ( size_t j = 0; j < digit_boxes.size(); j++)
		{
			cv::rectangle( lp, digit_boxes[j], cv::Scalar(50*j, 80*j, 255), 1, 8, 0);
			help::DrawText(lp_candidates[i].license_number.substr(j, 1), digit_boxes[j].tl(), lp);
		}

		cv::circle(lp, cv::Point(lp.cols/2, lp.rows/2), 2, cv::Scalar(255, 255, 0), -1);
		if (lp_candidates[i].box_long)
		{
			lp.copyTo(display(cv::Rect(0, config.text_box_resize*box_cnt, lp_candidates[i].lp_img.cols, lp_candidates[i].lp_img.rows)));
			box_cnt++;
		}
		else
		{
			lp.copyTo(display(cv::Rect(max_box_width, config.text_box_resize*long_cnt, lp_candidates[i].lp_img.cols, lp_candidates[i].lp_img.rows)));
			long_cnt++;
		}
	}

	cv::imshow("License Plate", display);
}

void LPR::LetterSpaceFilter(cv::Mat &data, int kernel_size, cv::Mat &dst)
{
	assert(data.rows == 1);
	assert(kernel_size % 2 == 1);
	if (data.type() != CV_32F)
		data.convertTo(data, CV_32F);

	dst = cv::Mat(data.size(), data.type());
	int tau = kernel_size/2;

	cv::Mat padding_data = cv::Mat::zeros(data.rows, data.cols + tau*2, data.type());
	data.copyTo(padding_data(cv::Rect(tau, 0, data.cols, data.rows)));

	float aux;
	const float *s = padding_data.ptr<float>(0);
	float *d = dst.ptr<float>(0);
	for ( int c = tau; c < data.cols + tau; c++ )
	{
		aux = 2*s[c];
		aux -= s[c - tau];
		aux -= s[c + tau];
		aux += std::abs( ( s[c - tau] - s[c + tau] ) );
		d[c - tau] = float(aux);
	}
}

void LPR::SaveLPs(const std::string &filename_template)
{

}

void LPR::FilterLPByColor(const cv::Mat &LP_img, cv::Mat &LP_crop_img)
{
	cv::Mat img, LP_resize_img, thres_frame;
	double ratio = 300.0/LP_img.rows;
	cv::resize(LP_img, LP_resize_img, cv::Size(), ratio, ratio);
	cv::cvtColor(LP_resize_img, img, CV_RGB2HSV);

	if (img.type() != CV_32F)
		img.convertTo(img, CV_32F);

	std::vector< cv::Mat> hsv_planes;
	cv::split( img, hsv_planes );

	for ( size_t i = 0; i < config.LP_hsv_lower_range.size(); i++)
	{
		cv::Mat temp, labels;
		cv::inRange(img, config.LP_hsv_lower_range[i], config.LP_hsv_upper_range[i], thres_frame);
		hsv_planes[2].copyTo(temp, thres_frame);
		std::cout << "Filter by " << i << std::endl;
		cv::imshow("LP filter by color", temp);
		cv::imshow("thres_frame", thres_frame);
		cv::Mat tmp = thres_frame.clone();
		Clustering(temp, thres_frame, labels);
//		labels = labels.reshape(1, hsv_planes[2].rows);
//		labels.convertTo(labels, CV_8U, 50);
		cv::imshow("labels", labels);

		cv::Mat best_LP;
		BestFitLP(labels, config.num_cols, config.num_cols, config.threshold, best_LP);
		cv::normalize(best_LP, best_LP, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );
		cv::imshow("best_LP", best_LP);

		std::vector< std::vector< cv::Point > > contours_poly;
		std::vector< std::vector< cv::Point > > contours;
		std::vector< cv::Vec4i > hierarchy;

//		tmp.convertTo(tmp, CV_8U);
		/// Find contours
		cv::findContours(tmp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		cv::cvtColor(tmp, tmp, CV_GRAY2RGB);
		for ( size_t j = 0; j < contours.size(); j++ )
			cv::drawContours(tmp, contours, j, cv::Scalar(0, 255, 255));
		cv::imshow("Contours", tmp);
		cv::waitKey(0);
	}
}

void LPR::Clustering(const cv::Mat &h_src, const cv::Mat &h_mask, cv::Mat &labels)
{
	cv::Mat img;
	h_src.copyTo(img, h_mask);
	if (img.type() != CV_32F)
		img.convertTo(img, CV_32F);

	int K = 3;
	cv::Mat centers;
	img = img.reshape(1, h_src.rows*h_src.cols);
	cv::kmeans(img, K, labels, cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
			3, cv::KMEANS_PP_CENTERS, centers);

	labels = labels.reshape(1, h_src.rows);
	labels.convertTo(labels, CV_8U, 50);

	labels += 50;

	int cnt = 0;
	int bg_label = 0;
	for ( int r = 0; r < h_mask.rows; r++ )
	{
		for ( int c = 0; c < h_mask.cols; c++ )
		{
			if (h_mask.at<uchar>(r, c) == 0)
			{
				if (bg_label == labels.at<uchar>(r, c))
				{
//					std::cout << "bg_label: " << bg_label << std::endl;
					cnt++;
				}
				else
					cnt = 0;
				bg_label = labels.at<uchar>(r, c);
				if (cnt == 3)
				{
					r = h_mask.rows;
					c = h_mask.cols;
				}
			}
		}
	}

//	std::cout << "final bg_label: " << bg_label << std::endl;

	cv::Mat labelLut = cv::Mat::zeros(1, 256, CV_8U);
	labelLut.at<char>(50) = 50;
	labelLut.at<char>(100) = 100;
	labelLut.at<char>(150) = 150;
	labelLut.at<char>(bg_label) = 0;
	cv::LUT(labels, labelLut, labels);

//	int label_values[] = {50, 100, 150};
//	std::vector< int > cnt_non_zero;
//	cnt_non_zero.resize(3);
//	tbb::parallel_for(0, 3, [&](int i) {
//		cv::Mat labelLut = cv::Mat::zeros(1, 256, CV_8U);
//		labelLut.at<char>(label_values[i]) = label_values[i];
//		cv::Mat temp_label;
//		cv::LUT(labels, labelLut, temp_label);
//		cnt_non_zero[i] = cv::countNonZero(temp_label);
//	});

//	cv::Mat kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
//	cv::morphologyEx(labels, labels, cv::MORPH_CLOSE, kernel);
//	cv::morphologyEx(labels, labels, cv::MORPH_OPEN, kernel);
//	cv::imshow("Morph", labels);
}

void LPR::BestFitLP(const cv::Mat &src, int ncol, int nrow, double thres, cv::Mat &best_LP)
{
	int w = src.cols/ncol;
	int h = src.rows/nrow;
	cv::Mat non_zero_cnt = cv::Mat::zeros(nrow, ncol, CV_32F);
	tbb::parallel_for(0, nrow, [&](int r) {
		for ( int c = 0; c < ncol; c++ )
		{
			int cnt = cv::countNonZero(src(cv::Rect(c*w, r*h, w, h)));
			if (cnt > (thres*w*h))
				non_zero_cnt.at<float>(r, c) = (float)cnt;
		}
	});

	double best_dense = -1;
	int best_s = nrow, best_r = 0, best_c = 0;
	for (int s = 2*nrow/3; s <= nrow; s++) {
		double dense;
		for ( int r = 0; r <= nrow - s; r++ )
		{
			for ( int c = 0; c <= ncol - s; c++ )
			{
				dense = (double)cv::sum(non_zero_cnt(cv::Rect(c, r, s, s)))[0]/(s*s);
				if (dense > best_dense)
				{
					best_dense = dense;
					best_s = s;
					best_r = r;
					best_c = c;
				}
			}
		}
	}

//	std::cout << "best_dense: " << best_dense << std::endl;
//	std::cout << "best_s: " << best_s << std::endl;
//	std::cout << "best_r: " << best_r << std::endl;
//	std::cout << "best_c: " << best_c << std::endl;
//	std::cout <<  non_zero_cnt << std::endl;


	if (best_s > 0)
	{
		int x = best_c*w;
		int y = best_r*h;
		cv::Rect best_fit_LP(x, y, std::min(src.cols - x - 1, w*best_s), std::min(src.rows - y - 1, h*best_s));
//		std::cout << "best_fit_LP: " << best_fit_LP << std::endl;
//		std::cout << "src: " << src.size() << std::endl;
		best_LP = src(best_fit_LP);
	}
	else
		best_LP = cv::Mat::zeros(src.size(), src.type());
}

std::string LPR::ClassifyDigits(LP_Candidate &candidate)
{
	cv::Mat result;
	cv::Mat img = candidate.lp_img.clone();
	if (img.channels() == 3)
		cv::cvtColor(img, img, CV_RGB2GRAY);

	help::ImAdjust(img, img, 5);
	std::vector< cv::Rect > digit_boxes = candidate.GetDigitBoxes();
//	cv::imshow("LP better contrast", img);
//	cv::waitKey(0);
	classifier.Classify(img, digit_boxes, result);
	return classifier.ClassIdx2String(result);
}

void LPR::FindLines(const bool box_long, std::vector< cv::Rect > &digit_boxes)
{
	// Get height median
	int height_median;
	std::sort(digit_boxes.begin(), digit_boxes.end(), BoxHeightCompare);
	if ((digit_boxes.size() % 2) == 1)
		height_median = digit_boxes[digit_boxes.size()/2].height;
	else
		height_median = (digit_boxes[digit_boxes.size()/2].height + digit_boxes[digit_boxes.size()/2 - 1].height)/2;

	// Long
	if (!box_long)
	{

	}
}

void LPR::FilterOutlierByHeight(std::vector< cv::Rect > &boxes)
{
	if (boxes.size() <= 3)
		return;

	// Filter by height
	std::sort(boxes.begin(), boxes.end(), BoxHeightCompare);
	std::vector< int > outlier_idx;
	for ( size_t i = 0; i < boxes.size(); i++ )
	{
		int median;
		if ((boxes.size() % 2) == 1)
		{
			if (i < boxes.size()/2)
				median = (boxes[boxes.size()/2].height + boxes[boxes.size()/2 + 1].height)/2;
			else if (i == boxes.size()/2)
				median = (boxes[boxes.size()/2 - 1].height + boxes[boxes.size()/2 + 1].height)/2;
			else
				median = (boxes[boxes.size()/2 - 1].height + boxes[boxes.size()/2].height)/2;
		}
		else
		{
			if (i < boxes.size()/2)
				median = boxes[boxes.size()/2].height;
			else
				median = boxes[boxes.size()/2 - 1].height;
		}

		double ratio = (double)boxes[i].height/median;
		if ((ratio < 0.8) || (ratio > 1.25))
			outlier_idx.push_back((int)i);
	}

	for ( int i = (int)outlier_idx.size() - 1; i >= 0; i-- )
		boxes.erase(boxes.begin() + outlier_idx[i]);
}

bool LPR::DigitCandidateTextLocCompare(Digit_Candidate first, Digit_Candidate second)
{
	if ((first.box.y + first.box.height/2) < second.box.y)
		return true;
	else if ((first.box.y + first.box.height/2) > (second.box.y + second.box.height))
		return false;
	else
		return (first.box.x < second.box.x);
}

bool LPTextLocCompare(cv::Rect first, cv::Rect second)
{
	if ((first.y + first.height/2) < second.y)
		return true;
	else if ((first.y + first.height/2) > (second.y + second.height))
		return false;
	else
		return (first.x < second.x);
}

bool BoxAreaCompare (cv::Rect first, cv::Rect second)
{
	return (first.area() < second.area());
}

bool BoxHeightCompare (cv::Rect first, cv::Rect second)
{
	return (first.height < second.height);
}

bool BoxLocXCompare (cv::Rect first, cv::Rect second)
{
	return (first.x < second.x);
}
