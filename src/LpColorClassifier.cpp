/*
 * LpColorClassifier.cpp
 *
 *  Created on: Jan 25, 2016
 *      Author: anhxtuan
 */

#include "HelpFnc.hpp"
#include "LpColorClassifier.hpp"

LpColor::LpColor()
{

}

LpColor::LpColor(const std::string &param_filename)
{
	Initialize(param_filename);
}

void LpColor::Initialize(const std::string &param_filename)
{
	params.read(param_filename);

	cvsvm = cv::ml::SVM::create();
	cvsvm->setType( cv::ml::SVM::C_SVC );
	cvsvm->setKernel( cv::ml::SVM::RBF );
	cvsvm->setTermCriteria( cv::TermCriteria( cv::TermCriteria::MAX_ITER, params.svm_max_iter, params.svm_epsilon));

	CvSvmLoadModel();
	LoadPCAcoeffs();
}

void LpColor::GetTrainingImagesFilename(const std::string input_dir, std::vector< std::vector< std::string > > &file_names, std::vector< std::string > &classes)
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
//							std::sort(fileNames.begin(), fileNames.end());
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

void LpColor::LoadImages(std::vector< std::vector< std::string > > &file_names, cv::Mat &dataset, cv::Mat &labels, cv::Mat &test_dataset, cv::Mat &test_labels)
{
	int num_train_data = 0, num_test_data = 0;
	std::vector< int > num_data;
	classWeight = cv::Mat(file_names.size(), 1, CV_32F);
	for ( size_t i = 0; i < file_names.size(); i++ )
	{
		int num_data_per_class = (int)(file_names[i].size()/(params.train_test_ratio + 1))*params.train_test_ratio;
		num_data_per_class = (num_data_per_class >  params.max_samples_class) ? params.max_samples_class : num_data_per_class;
		num_data.push_back(num_data_per_class);
		num_test_data += (std::min(num_data_per_class*2, (int)file_names[i].size()) - num_data_per_class);
		num_train_data += num_data_per_class;

		classWeight.at<float>(i) = 1.0 + ((float)params.max_samples_class/num_data_per_class - 1)*0.7;
	}

	std::cout << "Number of TEST data:  " << RED_TEXT << num_test_data << NORMAL_TEXT << std::endl;
	std::cout << "Number of TRAIN data: " << RED_TEXT << num_train_data << NORMAL_TEXT << std::endl;
	int k = 0, l = 0;
//	int num_features = params.image_size.area();
	int num_features = 256 + 256 + 180;
	dataset = cv::Mat(cv::Size(num_train_data, num_features), CV_32F);
	labels = cv::Mat(cv::Size(1, num_train_data), CV_32F);

	test_dataset = cv::Mat(cv::Size(num_test_data, num_features), CV_32F);
	test_labels = cv::Mat(cv::Size(1, num_test_data), CV_32F);

	for ( size_t i = 0; i < file_names.size(); i++ )
	{
//		bool wait = true;
		cv::Mat img;
		for ( int j = 0; j < std::min(num_data[i]*2, (int)file_names[i].size()); j++ )
		{
			img = cv::imread(file_names[i][j], CV_LOAD_IMAGE_UNCHANGED);
			ProcessImage(img);
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
		std::cout << "Finish load images of class: " << GREEN_TEXT << i << NORMAL_TEXT << std::endl;
		std::cout << "	- Num of TRAIN images: " << num_data[i] << std::endl;
		std::cout << "	- Num of TEST images:  " << (std::min(num_data[i]*2, (int)file_names[i].size()) - num_data[i]) << std::endl;
	}
	labels.convertTo(labels, CV_32S);
	test_labels.convertTo(test_labels, CV_32S);
}

/* This function will rearrange dataset for training in a random order. This step is
* necessary to make training more accurate.
*/
void LpColor::ShuffleDataset(cv::Mat &training_data, cv::Mat &label_mat, int numIter)
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

void LpColor::ProcessImage(cv::Mat &img)
{
	int x = 0.1*img.cols;
	int y = 0.1*img.rows;
	int w = 0.8*img.cols;
	int h = 0.8*img.rows;
	cv::Mat crop = img(cv::Rect(x, y, w, h)).clone();

	cv::resize(crop, crop, params.image_size);
	cv::GaussianBlur(crop, crop, cv::Size(0, 0), 9);
	cv::GaussianBlur(crop, crop, cv::Size(0, 0), 9);
	cv::imshow("lp color", crop);
	cv::cvtColor(crop, crop, CV_RGB2HSV);
	help::DoHist(crop, false);
	std::vector< cv::Mat > hsv_channels;
	cv::split(crop, hsv_channels);

	cv::Mat h_hist, s_hist, v_hist;
	help::DoHist(hsv_channels[0], h_hist);
	help::DoHist(hsv_channels[1], s_hist);
	help::DoHist(hsv_channels[2], v_hist);

	h_hist.convertTo(h_hist, CV_32F);
	s_hist.convertTo(s_hist, CV_32F);
	v_hist.convertTo(v_hist, CV_32F);

	int num_features = 256 + 256 + 180;
	img = cv::Mat(num_features, 1, CV_32F);

	cv::vconcat(h_hist.rowRange(0, 180), s_hist, img);
	cv::vconcat(img, v_hist, img);
}

void LpColor::PrepareDataset()
{
	std::vector< std::string > classes;
	cv::Mat dataset, labels, test_dataset, test_labels;
	std::vector< std::vector< std::string > > file_names;
	GetTrainingImagesFilename(params.input_dir, file_names, classes);
	LoadImages(file_names, dataset, labels, test_dataset, test_labels);
	ShuffleDataset(dataset, labels);

	for ( size_t i = 0; i < classes.size(); i++ )
	{
		std::cout << "Class: " << classes[i] << " - weight: " << classWeight.at<float>(i) << std::endl;
	}

	help::StoreDataset(params.trainset_bin_filename, dataset);
	help::StoreDataset(params.train_labels_bin_filename, labels);
	help::StoreDataset(params.testset_bin_filename, test_dataset);
	help::StoreDataset(params.test_labels_bin_filename, test_labels);
	help::StoreDataset(params.classes_bin_filename, classes);
}

void LpColor::PrepareDataset(std::vector< std::string > &classes, cv::Mat &dataset, cv::Mat &labels)
{
	cv::Mat test_dataset, test_labels;
	std::vector< std::vector< std::string > > file_names;
	GetTrainingImagesFilename(params.input_dir, file_names, classes);
	LoadImages(file_names, dataset, labels, test_dataset, test_labels);
	ShuffleDataset(dataset, labels);

	for ( size_t i = 0; i < classes.size(); i++ )
	{
		std::cout << "Class: " << classes[i] << " - weight: " << classWeight.at<float>(i)  << std::endl;
	}

	help::StoreDataset(params.trainset_bin_filename, dataset);
	help::StoreDataset(params.train_labels_bin_filename, labels);
	help::StoreDataset(params.testset_bin_filename, test_dataset);
	help::StoreDataset(params.test_labels_bin_filename, test_labels);
	help::StoreDataset(params.classes_bin_filename, classes);
}

void LpColor::GetTrainDataset(std::vector< std::string > &classes, cv::Mat &dataset, cv::Mat &labels)
{
	help::LoadDataset(params.trainset_bin_filename, dataset);
	help::LoadDataset(params.train_labels_bin_filename, labels);
	help::LoadDataset(params.classes_bin_filename, classes);
}

void LpColor::GetTestDataset(std::vector< std::string > &classes, cv::Mat &dataset, cv::Mat &labels)
{
	help::LoadDataset(params.testset_bin_filename, dataset);
	help::LoadDataset(params.test_labels_bin_filename, labels);
	help::LoadDataset(params.classes_bin_filename, classes);
}

void LpColor::PCAtransform(const cv::Mat &src, cv::Mat &mean, cv::Mat &coeffs, cv::Mat &dst)
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

	this->mean = mean;
	this->coeffs = coeffs;

	help::StoreDataset(params.pca_mean_bin_filename, mean);
	help::StoreDataset(params.pca_coeff_bin_filename, coeffs);
	help::StoreDataset(params.pca_trans_bin_filename, dst);
}

void LpColor::GetPCAtransform(cv::Mat &mean, cv::Mat &coeffs, cv::Mat &dst)
{
	help::LoadDataset(params.pca_mean_bin_filename, mean);
	help::LoadDataset(params.pca_coeff_bin_filename, coeffs);
	help::LoadDataset(params.pca_trans_bin_filename, dst);
}

void LpColor::CvSvmAutoTrain(const cv::Mat &src, const cv::Mat &labels)
{
	cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(src, cv::ml::SampleTypes::COL_SAMPLE, labels);
	cvsvm->setClassWeights(classWeight);
	cvsvm->trainAuto(tData, params.n_folds, params.cGrid, params.gammaGrid);
	cvsvm->save(params.svm_model_file_name);
}

void LpColor::CvSvmLoadModel()
{
	cvsvm = cv::ml::StatModel::load<cv::ml::SVM>(params.svm_model_file_name);
	std::cout << "Variable count: " << cvsvm->getVarCount() << std::endl;
	assert(cvsvm->getVarCount() > 0);
}

void LpColor::CvSvmPredict(const cv::Mat &src, cv::Mat &predict_labels)
{
	assert(cvsvm->isTrained() || cvsvm->isClassifier());
	assert(cvsvm->getVarCount() == src.rows);
	cvsvm->predict(src.t(), predict_labels);
}

void LpColor::LoadPCAcoeffs()
{
	help::LoadDataset(params.pca_coeff_bin_filename, coeffs);
	help::LoadDataset(params.pca_mean_bin_filename, mean);
	help::LoadDataset(params.classes_bin_filename, classes);
}

int LpColor::Classify(const cv::Mat &img)
{
	if (mean.empty())
		LoadPCAcoeffs();

	cv::Mat src = img.clone();
	ProcessImage(src);

	cv::Mat trans = coeffs.t()*src;
	return (int)cvsvm->predict(trans.t());
}

void LpColor::Classify(const std::vector< cv::Mat > &imgs, cv::Mat &result)
{
	if (mean.empty())
		LoadPCAcoeffs();

	cv::Mat src(params.image_size.area(), (int)imgs.size(), CV_32F); // row, col, type
	tbb::parallel_for(0, (int)imgs.size(), [&](int i) {
//	for ( size_t i = 0; i < imgs.size(); i++ ) {
		cv::Mat temp = imgs[i].clone();
		ProcessImage(temp);
		temp.copyTo(src.col(i));
	});

	cvsvm->predict(src.t(), result);
}

std::string LpColor::ClassIdx2String(cv::Mat &result)
{
	result.convertTo(result, CV_8U);
	std::string res = "";
	for (int r = 0; r < result.rows; r++)
	{
		res += classes[result.at<uchar>(r)];
	}
	return res;
}

double LpColor::Test(const cv::Mat &test_dataset, const cv::Mat &test_labels)
{
	if (mean.empty())
		LoadPCAcoeffs();

	cv::Mat src, avg, trans, pred_labels;
	trans = coeffs.t()*test_dataset;
	cvsvm->predict(trans.t(), pred_labels);
	pred_labels.convertTo(pred_labels, test_labels.type());
	return help::Accuracy(pred_labels, test_labels);
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
