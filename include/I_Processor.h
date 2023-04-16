// paul.straus@gmail.com
// all rights reserved
#pragma once


// file I_Processor.h

//Includes
#include <opencv2/opencv.hpp>

/// \brief abstract class describing what a Processor class must do
class I_Processor
{
  public:
    virtual ~I_Processor() = default;

    //actual functions
    virtual cv::Mat processNewImage(cv::Mat& newImage) = 0;
};