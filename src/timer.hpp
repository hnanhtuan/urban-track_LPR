// FILE: timer.hpp

#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <string>

// Only for timing sub-second duration events!
class timer {
    public:
        timer( std::string name = "timer" ) :
                    _name( name ), _count( 0 ), _sum_ns_diff( 0 ) {};
        ~timer( void ) {};

        void start( void );
        void stop( void );

        double diffs( void ) const;
        long   diffm( void ) const;
        long   diffu( void ) const;
        long   diffn( void ) const;

        void prints( void ) const;
        void printm( void ) const;
        void printu( void ) const;
        void printn( void ) const;

        long adiffm( void ) const;
        long adiffu( void ) const;
        long adiffn( void ) const;

        void aprintm( void ) const;
        void aprintu( void ) const;
        void aprintn( void ) const;

    private:
        std::string _name;
        long _ns_start;
        long _ns_stop;
        long _ns_diff;
        clock_t _clk_start;
        clock_t _clk_diff;

        unsigned long _count;
        unsigned long long _sum_ns_diff;

        unsigned long avg_ns_diff( void ) const;
};

#endif // ifndef _TIMER_HPP_



