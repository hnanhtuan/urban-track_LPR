/*
 * HelpFnc.hpp
 *
 *  Created on: Jan 4, 2016
 *      Author: anhxtuan
 */

#ifndef HELPFNC_HPP_
#define HELPFNC_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>      // std::setfill, std::setw

#include "def.hpp"

// Intel TBB
#include "tbb/tbb.h"

// Armadillo
#include <armadillo>

// Open CV
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

// Boost
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <boost/algorithm/string.hpp>

#include <boost/archive/tmpdir.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)

#define X_TRANSLATION	(0)
#define Y_TRANSLATION	(0)
#define Z_TRANSLATION	(380)
#define FOCAL_IN_PX		(378)

void PlotChart(std::vector< int > data, cv::Mat &chart);

void Padding(const cv::Mat &src, int pad_size_x, int pad_size_y, cv::Mat &dst);

void Rotate(const cv::Mat& src, double angle_radian, cv::Mat& dst);

void ImAdjust(const cv::Mat &src, cv::Mat &dst, int tol = 1);

void ImAdjust(const cv::Mat &src, cv::Mat &dst, cv::Vec2i in = cv::Vec2i(0, 255));

void DoHist(const cv::Mat &img, bool wait = false);

void CreateDirectory(const std::string &path);

void CopyNonZero(cv::Mat &src, cv::Mat &dst);

void GetTrainingImages(const std::string input_dir, std::vector< std::vector< std::string > > &file_names, std::vector< std::string > &classes);

void SaveImage(const cv::Mat &img, const std::string file_name_format, int file_num = -1);

void DrawBoxes(const cv::Mat &src, const std::vector< cv::Rect > boxes, cv::Mat &dst);

void DrawBoxesAndShow(cv::Mat &src, std::vector< cv::Rect > boxes, std::string win_name = "box", bool wait = false);

bool IsExist(const std::string &filename);

void RemoveFiles(const std::string &filename);

const std::string CurrentDateTime();

void StoreDataset(const std::string &filename, const cv::Mat& m);

void StoreDataset(const std::string &filename, const std::vector< std::string > &vec);

void LoadDataset(const std::string &filename, cv::Mat& m);

void LoadDataset(const std::string &filename, std::vector< std::string > &vec);

void CopyFile(const std::string &src_file, const std::string &dst_file);

void Rotate3D(const cv::Mat &img, cv::Mat &dst, float theta, float gamma, float beta);

template<typename T>
void Cv_mat_to_arma_mat(const cv::Mat_<T>& cv_mat_in, arma::Mat<T>& arma_mat_out);

template<typename T>
void Arma_mat_to_cv_mat(const arma::Mat<T>& arma_mat_in, cv::Mat_<T>& cv_mat_out);

#endif /* HELPFNC_HPP_ */
