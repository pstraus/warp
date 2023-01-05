// paul.straus@gmail.com
// all rights reserved

// file I_ReferenceImage.h

// Includes
#include <ExponentialAverage.h>
#include <stdexcept>
#include <opencv2/cudaarithm.hpp>

/// \brief abstract class describing what a reference image must do /
using namespace cv;

ExponentialAverage::ExponentialAverage(double alpha)
: m_initialized{false}, m_alpha{alpha}
{}

ExponentialAverage::ExponentialAverage(const cuda::GpuMat& firstImage, double alpha)
    : m_ref{firstImage}, m_initialized(true), m_alpha(alpha)
{
  // Nothing to do
}

// actual functions
const cuda::GpuMat ExponentialAverage::getReferenceImage() const
{
  return this->m_ref;
}
void ExponentialAverage::update(const cv::cuda::GpuMat& newImage)
{
  double beta = 1.0 - m_alpha;
  std::cout << "m_initialized is: " <<m_initialized << "\t m_alpha is: " << m_alpha << std::endl;

  cv::Mat testImage;
  m_ref.download(testImage);
  cv::imshow("reference", testImage);
  cv::waitKey(25);
  if(this->m_initialized){
    cuda::GpuMat tmp(this->m_ref);
    cuda::addWeighted(tmp, m_alpha, newImage, beta, 0.0, this->m_ref, -1, cuda::Stream::Null());
  }
  else{
  this->m_ref = newImage;
  this->m_initialized = true;
  }
}
bool ExponentialAverage::valid() const
{
  return this->m_initialized;
}
