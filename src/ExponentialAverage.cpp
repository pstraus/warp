// paul.straus@gmail.com
// all rights reserved

// file I_ReferenceImage.h

// Includes
#include <ExponentialAverage.h>
#include <stdexcept>
#include <opencv2/cudaarithm.hpp>

/// \brief abstract class describing what a reference image must do /
using namespace cv;
using namespace cv::cuda;

ExponentialAverage::ExponentialAverage(const GpuMat &firstImage, const double alpha)
    : m_ref{firstImage}, m_initialized(true), m_alpha(alpha)
{
  // Nothing to do
}

// actual functions
const GpuMat &ExponentialAverage::getReferenceImage() const
{
  return this->m_ref;
}
void ExponentialAverage::update(const cv::cuda::GpuMat &newImage)
{
  if(this->m_initialized){
    addWeighted(this->m_ref, m_alpha, newImage, 1.0-m_alpha, 0.0, this->m_ref, -1, Stream::Null());
  }
  else{
    std::string error("attempting to access reference image before it is valid or initialized!");
    throw(std::logic_error(error));
  }
}
bool ExponentialAverage::valid() const
{
  return this->m_initialized;
}
