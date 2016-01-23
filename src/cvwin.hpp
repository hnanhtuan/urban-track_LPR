// FILE: cvwin.hpp

#ifndef _CVWIN_HPP_
#define _CVWIN_HPP_

#include "opencv2/highgui/highgui.hpp"
#include <string>

class cvwin {
    public:
        cvwin( std::string name = "win" );
        ~cvwin( void );

        void display_frame( const cv::Mat &frame ) const;

    protected:
        std::string name;
};

#endif // ifndef _CVWIN_HPP_



