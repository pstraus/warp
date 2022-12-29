// paul.straus@gmail.com
// all rights reserved

// file I_ReferenceImage.h

// Includes
#include <warp/ExponentialAverage.h>

/// \brief abstract class describing what a reference image must do /

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
void ExponentialAverage::update(const cv::Mat &newImage)
{
  addWeighted(this->m_ref, m_alpha, newImage, 1-m_alpha, 0.0, this->m_ref);
}
bool ExponentialAverage::valid() const
{
  return this->m_initialized;
}
