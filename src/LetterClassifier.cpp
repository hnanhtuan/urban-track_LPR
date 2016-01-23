/*
 * LetterClassifier.cpp
 *
 *  Created on: Jan 5, 2016
 *      Author: anhxtuan
 */


#include "LetterClassifier.hpp"

LetterClassifier::LetterClassifier()
{

}

LetterClassifier::LetterClassifier(const std::string &param_filename)
{
	Initialize(param_filename);
}

void LetterClassifier::Initialize(const std::string &param_filename)
{
	params.read(param_filename);

	cvsvm = cv::ml::SVM::create();
	cvsvm->setType( cv::ml::SVM::C_SVC );
	cvsvm->setKernel( cv::ml::SVM::RBF );
	cvsvm->setTermCriteria( cv::TermCriteria( cv::TermCriteria::MAX_ITER, params.svm_max_iter, params.svm_epsilon));


	if (params.phase == PREPARE)
	{
//		cv::Mat dataset, labels, trans;
//		PrepareDataset(dataset, labels);
//		PCAtransform(dataset, mean, coeffs, trans);
	}
	else if (params.phase == CLASSIFY)
	{
		CvSvmLoadModel();
		LoadPCAcoeffs();
	};
}

void LetterClassifier::GetTrainingImagesFilename(const std::string input_dir, std::vector< std::vector< std::string > > &file_names, std::vector< std::string > &classes)
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

void LetterClassifier::LoadImages(std::vector< std::vector< std::string > > &file_names, cv::Mat &dataset, cv::Mat &labels, cv::Mat &test_dataset, cv::Mat &test_labels)
{
	int num_train_data = 0, num_test_data = 0;
	std::vector< int > num_data;
	for ( size_t i = 0; i < file_names.size(); i++ )
	{
		int num_data_per_class = (int)(file_names[i].size()/(params.train_test_ratio + 1))*params.train_test_ratio;
		num_data_per_class = (num_data_per_class >  params.max_samples_class) ? params.max_samples_class : num_data_per_class;
		num_data.push_back(num_data_per_class);
		num_test_data += ((int)file_names[i].size() - num_data_per_class);
		num_train_data += num_data_per_class;
	}
	std::cout << "Number of TEST data:  " << num_test_data << std::endl;
	std::cout << "Number of TRAIN data: " << num_train_data << std::endl;
	int k = 0, l = 0;
	dataset = cv::Mat(cv::Size(num_train_data, params.letter_size.area()), CV_32F);
	labels = cv::Mat(cv::Size(1, num_train_data), CV_32F);

	test_dataset = cv::Mat(cv::Size(num_test_data, params.letter_size.area()), CV_32F);
	test_labels = cv::Mat(cv::Size(1, num_test_data), CV_32F);
	for ( size_t i = 0; i < file_names.size(); i++ )
	{
		cv::Mat img;
		for ( int j = 0; j < (int)file_names[i].size(); j++ )
		{
			img = cv::imread(file_names[i][j], CV_LOAD_IMAGE_GRAYSCALE);
			cv::resize(img, img, params.letter_size);
			img.convertTo(img, CV_32F);
			img = img.reshape(1, params.letter_size.area());
			if (j < num_data[i])
			{
				img.copyTo(dataset.col(k));
				labels.at<float>(k) = (float)i;
				k++;
			}
			else
			{
				img.copyTo(test_dataset.col(l));
				test_labels.at<float>(l) = (float)i;
				l++;
			}
		}
	}
	labels.convertTo(labels, CV_32S);
	test_labels.convertTo(test_labels, CV_32S);
}

/* This function will rearrange dataset for training in a random order. This step is
* necessary to make training more accurate.
*/
void LetterClassifier::ShuffleDataset(cv::Mat &training_data, cv::Mat &label_mat, int numIter)
{
	/* initialize random seed: */
	srand(time(NULL));
	int x = 0, y = 0;

	assert(training_data.cols == label_mat.rows);

	int numData = training_data.cols;
	if (numIter <= 0)
		numIter = numData;

	if (training_data.type() != CV_32FC1)
		training_data.convertTo(training_data, CV_32FC1);
	cv::Mat temp_data_mat(training_data.rows, 1, CV_32FC1);
	cv::Mat temp_label_mat(1, 1, CV_32FC1);


	// Interate 'numIter' to rearrange dataset
	for (int n = 0; n < numIter; n++)
	{
		x = (rand() % numData);
		y = (rand() % numData);

		// swap data
		training_data.col(x).copyTo(temp_data_mat.col(0));
		training_data.col(y).copyTo(training_data.col(x));
		temp_data_mat.col(0).copyTo(training_data.col(y));

		// swap label
		label_mat.row(x).copyTo(temp_label_mat.row(0));
		label_mat.row(y).copyTo(label_mat.row(x));
		temp_label_mat.row(0).copyTo(label_mat.row(y));
	}
}

void LetterClassifier::PrepareDataset()
{
	std::vector< std::string > classes;
	cv::Mat dataset, labels, test_dataset, test_labels;
	std::vector< std::vector< std::string > > file_names;
	GetTrainingImagesFilename(params.input_dir, file_names, classes);
	LoadImages(file_names, dataset, labels, test_dataset, test_labels);
	ShuffleDataset(dataset, labels);

	StoreDataset(params.dataset_bin_filename, dataset);
	StoreDataset(params.labels_bin_filename, labels);
	StoreDataset(params.test_dataset_bin_filename, test_dataset);
	StoreDataset(params.test_labels_bin_filename, test_labels);
	StoreDataset(params.classes_bin_filename, classes);
}

void LetterClassifier::PrepareDataset(std::vector< std::string > classes, cv::Mat dataset, cv::Mat labels)
{
	cv::Mat test_dataset, test_labels;
	std::vector< std::vector< std::string > > file_names;
	GetTrainingImagesFilename(params.input_dir, file_names, classes);
	LoadImages(file_names, dataset, labels, test_dataset, test_labels);
	ShuffleDataset(dataset, labels);

	StoreDataset(params.dataset_bin_filename, dataset);
	StoreDataset(params.labels_bin_filename, labels);
	StoreDataset(params.test_dataset_bin_filename, test_dataset);
	StoreDataset(params.test_labels_bin_filename, test_labels);
	StoreDataset(params.classes_bin_filename, classes);
}

void LetterClassifier::GetTrainDataset(std::vector< std::string > &classes, cv::Mat &dataset, cv::Mat &labels)
{
	LoadDataset(params.dataset_bin_filename, dataset);
	LoadDataset(params.labels_bin_filename, labels);
	LoadDataset(params.classes_bin_filename, classes);
}

void LetterClassifier::GetTestDataset(std::vector< std::string > &classes, cv::Mat &dataset, cv::Mat &labels)
{
	LoadDataset(params.test_dataset_bin_filename, dataset);
	LoadDataset(params.test_labels_bin_filename, labels);
	LoadDataset(params.classes_bin_filename, classes);
}

void LetterClassifier::PCAtransform(const cv::Mat &src, cv::Mat &mean, cv::Mat &coeffs, cv::Mat &dst)
{
	cv::Mat avg;
	int num_samples = src.cols;
	cv::reduce(src, mean, 1, CV_REDUCE_AVG);	// reduce to single column

	cv::repeat(mean, 1, num_samples, avg);
	cv::Mat x = src - avg;
	cv::Mat sigma = (x*x.t())/num_samples;
	sigma.convertTo(sigma, CV_64F);

	// Using Armadillo library for SVD compute
	arma::Mat<double> armaSigma;
	arma::Mat<double> armaU, armaV, armaS;
	arma::vec armas;
	cv::Mat_<double> w, u, vt;

	// convert from CV Mat to Arma Mat
	Cv_mat_to_arma_mat<double>(sigma, armaSigma);

	// execute SVD
	arma::svd(armaU, armas, armaV, armaSigma);

	// convert from Arma Mat back to CV Mat
	Arma_mat_to_cv_mat<double>(armaU, u);
//	Arma_mat_to_cv_mat<double>(armaV, vt);
//	armaS = arma::conv_to< arma::Mat<double> >::from(armas);
//	Arma_mat_to_cv_mat<double>(armaS, w);

	if (params.pca_max_comp == 0)
	{
		double denom = arma::sum(armas);
		double numer = denom;
		for ( int i = (int)armas.n_rows - 1; i >= 0; i-- )
		{
			numer -= armas[i];
			double perVarRetain = numer/denom;
			if (perVarRetain < params.pca_ret_var)
			{
				params.pca_max_comp = i + 1;
				break;
			}
		}
	}
	else
	{
		params.pca_max_comp = std::min(params.pca_max_comp, (int)armas.n_rows);
		double denom = arma::sum(armas);
		double numer = 0;
		for ( int i = 0; i < params.pca_max_comp; i++ )
		{
			numer += armas[i];
		}
		params.pca_ret_var = numer/denom;
	}
	coeffs = u(cv::Rect(0, 0, params.pca_max_comp, u.rows));
	if (coeffs.type() != src.type())
		coeffs.convertTo(coeffs, src.type());
	dst = coeffs.t()*src;

	StoreDataset(params.pca_mean_bin_filename, mean);
	StoreDataset(params.pca_coeff_bin_filename, coeffs);
	StoreDataset(params.pca_trans_bin_filename, dst);

	this->mean = mean;
	this->coeffs = coeffs;
}

void LetterClassifier::GetPCAtransform(cv::Mat &mean, cv::Mat &coeffs, cv::Mat &dst)
{
	LoadDataset(params.pca_mean_bin_filename, mean);
	LoadDataset(params.pca_coeff_bin_filename, coeffs);
	LoadDataset(params.pca_trans_bin_filename, dst);
}

void LetterClassifier::CvSvmAutoTrain(const cv::Mat &src, const cv::Mat &labels)
{
	cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(src, cv::ml::SampleTypes::COL_SAMPLE, labels);

	cvsvm->trainAuto(tData, params.n_folds, params.cGrid, params.gammaGrid);
	cvsvm->save(params.svm_model_file_name);
}

void LetterClassifier::CvSvmLoadModel()
{
	cvsvm = cv::ml::StatModel::load<cv::ml::SVM>(params.svm_model_file_name);
//	cvsvm->load<cv::ml::SVM>(params.svm_model_file_name);
	std::cout << "cv SVM variable count: " << cvsvm->getVarCount() << std::endl;
	assert(cvsvm->getVarCount() > 0);
}

void LetterClassifier::CvSvmPredict(const cv::Mat &src, cv::Mat &predict_labels)
{
	assert(cvsvm->isTrained() || cvsvm->isClassifier());
	assert(cvsvm->getVarCount() == src.rows);
	cvsvm->predict(src.t(), predict_labels);
}

double LetterClassifier::Accuracy(const cv::Mat &labels, const cv::Mat &predicts)
{
	assert(labels.size() == predicts.size());
	assert(labels.type() == predicts.type());

	cv::Mat accuracy;
	cv::compare(labels, predicts, accuracy, CV_CMP_EQ);
	return (double)cv::countNonZero(accuracy)/accuracy.size().area();
}

void LetterClassifier::LoadPCAcoeffs()
{
	LoadDataset(params.pca_coeff_bin_filename, coeffs);
	LoadDataset(params.pca_mean_bin_filename, mean);
	LoadDataset(params.classes_bin_filename, classes);
}

int LetterClassifier::Classify(const cv::Mat &img)
{
	if (mean.empty())
		LoadPCAcoeffs();

	cv::Mat src;
	if (img.channels() == 3)
		cv::cvtColor(img, src, CV_RGB2GRAY);
	else
		img.copyTo(src);

	if (src.size() != params.letter_size)
		cv::resize(src, src, params.letter_size);

	cv::Mat blur_img;
	cv::GaussianBlur(src, blur_img, cv::Size(0, 0), 5);
	cv::addWeighted(src, 4, blur_img, -3, 0, src);

	if (src.type() != CV_32F)
		src.convertTo(src, CV_32F);

	src = src.reshape(1, params.letter_size.area());
	cv::Mat trans = coeffs.t()*src;
	return (int)cvsvm->predict(trans.t());
}

void LetterClassifier::Classify(const std::vector< cv::Mat > &imgs, cv::Mat &result)
{
	if (imgs.size() == 0)
		return;

	if (cvsvm->getVarCount() == 0)
		CvSvmLoadModel();

	if (mean.empty())
		LoadPCAcoeffs();

	cv::Mat src(params.letter_size.area(), (int)imgs.size(), CV_32F); // row, col, type
	tbb::parallel_for(0, (int)imgs.size(), [&](int i) {
//	for ( size_t i = 0; i < imgs.size(); i++ ) {
		cv::Mat temp;
		if (imgs[i].channels() == 3)
			cv::cvtColor(imgs[i], temp, CV_RGB2GRAY);
		else
			imgs[i].copyTo(temp);

		if (temp.size() != params.letter_size)
			cv::resize(temp, temp, params.letter_size);

		cv::Mat blur_img;
		cv::GaussianBlur(temp, blur_img, cv::Size(0, 0), 5);
		cv::addWeighted(temp, 4, blur_img, -3, 0, temp);

		if (temp.type() != CV_32F)
			temp.convertTo(temp, CV_32F);

		temp = temp.reshape(1, params.letter_size.area());
		temp.copyTo(src.col(i));
	});

	cv::Mat trans = coeffs.t()*src;
	cvsvm->predict(trans.t(), result);
}

std::string LetterClassifier::ClassIdx2String(cv::Mat &result)
{
	result.convertTo(result, CV_8U);
	std::string res = "";
	for (int r = 0; r < result.rows; r++)
	{
		res += classes[result.at<uchar>(r)];
	}
	return res;
}

void LetterClassifier::Classify(const cv::Mat &img, const std::vector< cv::Rect > &boxes, cv::Mat &result)
{
	if (boxes.size() == 0)
		return;

	if (cvsvm->getVarCount() == 0)
		CvSvmLoadModel();

	if (mean.empty())
		LoadPCAcoeffs();

	cv::Mat src(params.letter_size.area(), (int)boxes.size(), CV_32F); // row, col, type
	tbb::parallel_for(0, (int)boxes.size(), [&](int i) {
//	for ( size_t i = 0; i < boxes.size(); i++ ) {
		cv::Mat temp = img(boxes[i]);
		if (temp.channels() == 3)
			cv::cvtColor(temp, temp, CV_RGB2GRAY);

		if (temp.size() != params.letter_size)
			cv::resize(temp, temp, params.letter_size);

		cv::Mat blur_img;
		cv::GaussianBlur(temp, blur_img, cv::Size(0, 0), 5);
		cv::addWeighted(temp, 4, blur_img, -3, 0, temp);

//		cv::Mat temp_inv;
//		cv::bitwise_not(temp, temp_inv);
//		cv::imshow("inv", temp_inv);
//		cv::waitKey(0);

		if (temp.type() != CV_32F)
		{
			temp.convertTo(temp, CV_32F);
//			temp_inv.convertTo(temp_inv, CV_32F);
		}

		temp = temp.reshape(1, params.letter_size.area());
		temp.copyTo(src.col(i));

//		temp_inv = temp_inv.reshape(1, params.letter_size.area());
//		temp_inv.copyTo(src.col(i*2 + 1));
	});
	cv::Mat trans = coeffs.t()*src;
	cvsvm->predict(trans.t(), result);
}

double LetterClassifier::Test(const cv::Mat &test_dataset, const cv::Mat &test_labels)
{
	if (mean.empty())
		LoadPCAcoeffs();

	cv::Mat src, avg, trans, pred_labels;
//	cv::repeat(mean, 1, test_dataset.cols, avg);
//	src = test_dataset - avg;
	trans = coeffs.t()*test_dataset;
	cvsvm->predict(trans.t(), pred_labels);
	pred_labels.convertTo(pred_labels, test_labels.type());
	return Accuracy(pred_labels, test_labels);
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
