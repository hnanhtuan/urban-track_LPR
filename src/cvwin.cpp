// FILE: cvwin.cpp

#include "cvwin.hpp"

cvwin::cvwin( std::string name ) : name(name) {
    cv::namedWindow( name, CV_WINDOW_AUTOSIZE );
}

cvwin::~cvwin( void ) {
    cv::destroyWindow( name );
}

void cvwin::display_frame( const cv::Mat &frame ) const {
    cv::imshow( name, frame );
}



