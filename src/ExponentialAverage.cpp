// paul.straus@gmail.com
// all rights reserved

// file ExponentialAverage.cpp

// Includes
#include <ExponentialAverage.h>
#include <stdexcept>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

/// \brief Concrete implementation 
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
const cuda::GpuMat& ExponentialAverage::getReferenceImage() const
{
  return this->m_ref;
}
void ExponentialAverage::update(const cv::cuda::GpuMat& newImage)
{
  if(this->m_initialized){
    cuda::GpuMat newReference;   
    cuda::addWeighted(m_ref, m_alpha, newImage, 1.0-m_alpha, 0.0, newReference);

    m_ref = newReference;
  }
  else{
    this->m_initialized = true;
  }
  
}
bool ExponentialAverage::valid() const
{
  return this->m_initialized;
}
