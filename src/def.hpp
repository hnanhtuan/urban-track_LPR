/*
 * def.hpp
 *
 *  Created on: Jan 10, 2016
 *      Author: anhxtuan
 */

#ifndef DEF_HPP_
#define DEF_HPP_

#include <assert.h>
#include <string>
#include <opencv2/core/core.hpp>

#define RED_TEXT			( "\033[22;31m" )
#define NORMAL_TEXT			( "\e[m" )
#define GREEN_TEXT			( "\e[0;32m" )
#define YELLOW_TEXT			( "\e[0;33m" )
#define BLUE_TEXT			( "\e[0;34m" )

extern int DEBUG_LEVEL;

#define DEBUG

#ifdef DEBUG
#define DMESG( v, x ) 	{ if (x >= DEBUG_LEVEL) { std::cout << v << std::endl; } }
#else
#define DMESG( v, x )	do { } while ( false )
#endif

#endif /* DEF_HPP_ */
