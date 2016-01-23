/*
 * LetterClassifier.hpp
 *
 *  Created on: Jan 5, 2016
 *      Author: anhxtuan
 */

#ifndef LETTERCLASSIFIER_HPP_
#define LETTERCLASSIFIER_HPP_

#include <algorithm>    // std::sort
#include <vector>       // std::vector

#include "HelpFnc.hpp"

// Intel TBB
#include "tbb/tbb.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

class LetterClassifier {
private:
	typedef enum Phase {
		PREPARE,
		PCA,
		TRAIN,
		TEST,
		CLASSIFY
	} Phase;
	typedef struct LetterClassifierParam {

		Phase phase;
		std::string input_dir;
		cv::Size letter_size;
		int max_samples_class;
		double train_test_ratio;

		std::string svm_model_file_name;
		std::string dataset_bin_filename, labels_bin_filename, classes_bin_filename;
		std::string test_dataset_bin_filename, test_labels_bin_filename;
		std::string pca_mean_bin_filename, pca_coeff_bin_filename, pca_trans_bin_filename;

		int pca_max_comp;
		double pca_ret_var;

		int svm_max_iter, n_folds;
		double svm_epsilon;

		cv::ml::ParamGrid cGrid, gammaGrid;

		void read(const std::string &filename)
		{
			cv::FileStorage fs(filename, cv::FileStorage::READ);

			int phase_val = (int)fs["phase"];
			phase = static_cast<Phase>(phase_val);
			fs["input_dir"] >> input_dir;
			fs["letter_size"] >> letter_size;

			max_samples_class = (int)fs["max_samples_class"];
			train_test_ratio = (double)fs["train_test_ratio"];

			fs["dataset_bin_filename"] >> dataset_bin_filename;
			fs["labels_bin_filename"] >> labels_bin_filename;
			fs["test_dataset_bin_filename"] >> test_dataset_bin_filename;
			fs["test_labels_bin_filename"] >> test_labels_bin_filename;
			fs["classes_bin_filename"] >> classes_bin_filename;

			fs["pca_mean_bin_filename"] >> pca_mean_bin_filename;
			fs["pca_coeff_bin_filename"] >> pca_coeff_bin_filename;
			fs["pca_trans_bin_filename"] >> pca_trans_bin_filename;

			fs["svm_model_file_name"] >> svm_model_file_name;

			pca_max_comp = (int)fs["pca_max_comp"];
			pca_ret_var = (double)fs["pca_ret_var"];

			svm_max_iter = (int)fs["svm_max_iter"];
			svm_epsilon = (double)fs["svm_epsilon"];
			n_folds = (int)fs["n_folds"];

			cGrid.minVal = (double)fs["c_grid_min_val"];
			cGrid.maxVal = (double)fs["c_grid_max_val"];
			cGrid.logStep = (double)fs["c_grid_log_step"];

			gammaGrid.minVal = (double)fs["gamma_grid_min_val"];
			gammaGrid.maxVal = (double)fs["gamma_grid_max_val"];
			gammaGrid.logStep = (double)fs["gamma_grid_log_step"];
			fs.release();
		}
		void write(const std::string &filename)
		{
			cv::FileStorage fs(filename, cv::FileStorage::WRITE);
			phase = CLASSIFY;
			fs << "phase" << phase;
			fs.release();
		}
	} LetterClassifierParam;

	LetterClassifierParam params;

	cv::Ptr< cv::ml::SVM > cvsvm;

	void GetTrainingImagesFilename(const std::string input_dir, std::vector< std::vector< std::string > > &file_names, std::vector< std::string > &classes);

	void LoadImages(std::vector< std::vector< std::string > > &file_names, cv::Mat &dataset, cv::Mat &labels, cv::Mat &test_dataset, cv::Mat &test_labels);

	void ShuffleDataset(cv::Mat &training_data, cv::Mat &label_mat, int numIter = 500);
public:
	cv::Mat mean, coeffs;
	std::vector< std::string > classes;
	LetterClassifier();

	LetterClassifier(const std::string &param_filename);

	void Initialize(const std::string &param_filename);

	void PrepareDataset();

	void PrepareDataset(std::vector< std::string > classes, cv::Mat dataset, cv::Mat labels);

	void GetTrainDataset(std::vector< std::string > &classes, cv::Mat &dataset, cv::Mat &labels);

	void GetTestDataset(std::vector< std::string > &classes, cv::Mat &dataset, cv::Mat &labels);

	void PCAtransform(const cv::Mat &src, cv::Mat &mean, cv::Mat &coeffs, cv::Mat &dst);

	void GetPCAtransform(cv::Mat &mean, cv::Mat &coeffs, cv::Mat &dst);

	void CvSvmAutoTrain(const cv::Mat &src, const cv::Mat &labels);

	void CvSvmPredict(const cv::Mat &src, cv::Mat &predict_labels);

	void CvSvmLoadModel();

	void LoadPCAcoeffs();

	int Classify(const cv::Mat &img);

	void Classify(const std::vector< cv::Mat > &imgs, cv::Mat &result);

	void Classify(const cv::Mat &img, const std::vector< cv::Rect > &boxes, cv::Mat &result);

	std::string ClassIdx2String(cv::Mat &result);

	double Test(const cv::Mat &test_dataset, const cv::Mat &test_labels);

	double Accuracy(const cv::Mat &labels, const cv::Mat &predicts);

	double F1Value(const cv::Mat &labels, const cv::Mat &predicts);
};

template<typename T>
void Cv_mat_to_arma_mat(const cv::Mat_<T>& cv_mat_in, arma::Mat<T>& arma_mat_out);

template<typename T>
void Arma_mat_to_cv_mat(const arma::Mat<T>& arma_mat_in, cv::Mat_<T>& cv_mat_out);



#endif /* LETTERCLASSIFIER_HPP_ */
