// paul.straus@gmail.com
// all rights reserved

// file WarpingProcessor.cpp

//Includes
#include <WarpingProcessor.h>
#include <memory>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <stdexcept>


using namespace cv;
//Constructor consumes unique_ptr of type I_ReferenceImage
/// \note: explicitly specify cv::cuda::FarnebackOpticalFlow as there is a name collision with cv::FarnebackOpticalFlow
WarpingProcessor::WarpingProcessor(std::unique_ptr<I_ReferenceImage>&& refImage, Ptr<cuda::FarnebackOpticalFlow> flowPtr)
: m_ref{std::move(refImage)}, mp_opticalFlow{flowPtr}, m_x{}, m_y{}
{
  if(m_ref->valid())
  {
   genBaseCoordinateMaps(m_ref->getReferenceImage());
  }
  //Verify class is initialized properly
  //if(this->m_ref.get() == nullptr)
  //{
  //  std::string exception("Attempting to initialize WarpingProcessor with uninitialized reference image!");
  //  throw(std::logic_error(exception));
  //}

  //if(mp_opticalFlow.get() == nullptr)
  //{
  // std::string exception("Initialized WarpingProcessor using uninitialized Optical Flow Object!");
   //throw(std::logic_error(exception)); 
  //}
}


//actual functions
cuda::GpuMat WarpingProcessor::processNewImage(cv::cuda::GpuMat& newImage)
{
  //Allocate 
  cuda::GpuMat processedImage(newImage.size(), newImage.type());
  cuda::GpuMat grayScaleImage;
  cuda::GpuMat grayRef;


  if(m_ref->valid()){
    const auto& imgSize = this->m_ref->getReferenceImage().size();
   
    cuda::GpuMat grayScaleImage(imgSize, CV_32FC1);
    cuda::GpuMat grayScaleImage_smooth(imgSize, CV_32FC1);
    cuda::GpuMat grayRef(imgSize, CV_32FC1);
 
    //Compute the optical flow relative to the reference image
    cuda::GpuMat flowImage(imgSize, CV_32FC2);
    cuda::GpuMat flowImage_smooth(imgSize, CV_32FC2);
 
    cuda::cvtColor(newImage, grayScaleImage, COLOR_RGB2GRAY, 1, cuda::Stream::Null());
    cuda::GpuMat tmp(this->m_ref->getReferenceImage());//const correctnesss...
    cuda::cvtColor(tmp, grayRef, COLOR_RGB2GRAY, 1, cuda::Stream::Null());
 
    //estimate
    mp_opticalFlow->calc(grayScaleImage, grayRef, flowImage, cv::cuda::Stream::Null());
    
    //Average all flow vectors to estimate smoothing required
    auto averageShift = cv::cuda::absSum(flowImage) / (flowImage.size().height * flowImage.size().width);
    
    constexpr uint filterWidth = 11;
    constexpr uint filterHeight = 11;
    cv::Size filterSize( filterWidth, filterHeight);
    static_assert(filterWidth %2 != 0 && filterHeight %2 != 0, "Filter width and height must be odd!");

    // Clearly there is an optimization to not generating the filter on every frame.  But this will work for now (it's really just a kernel generation)
    auto filter = cv::cuda::createGaussianFilter(grayScaleImage.type(), grayScaleImage_smooth.type(), filterSize, averageShift[0], averageShift[1]);

    filter->apply(grayScaleImage, grayScaleImage_smooth);

    mp_opticalFlow->calc(grayScaleImage_smooth, grayRef, flowImage_smooth, cv::cuda::Stream::Null());   
    mp_opticalFlow->calc(grayRef, grayScaleImage, flowImage, cv::cuda::Stream::Null());   
      
    cv::Size flowFilterSize(5,5);
    auto flowFilter = cuda::createBoxFilter(CV_32FC1,CV_32FC1,flowFilterSize);

    cuda::GpuMat chans[2];
    cuda::split(flowImage, chans);

    cuda::GpuMat x_map_smooth(chans[0]);
    cuda::GpuMat y_map_smooth(chans[1]);

    flowFilter->apply(chans[0], x_map_smooth);
    flowFilter->apply(chans[1], y_map_smooth);

    cuda::GpuMat x_map(m_x.size(), m_x.type());
    cuda::GpuMat y_map(m_y.size(), m_y.type());
 
    cuda::add(m_x, x_map_smooth, x_map);
    cuda::add(m_y, y_map_smooth, y_map);
 
    //Warp the newImage based on the optical flow
    cuda::remap(newImage, processedImage, x_map, y_map, INTER_LINEAR, BORDER_REPLICATE, 0, cuda::Stream::Null());
 
    //Update the reference Image using the new image (yes, this should be done AFTER)
    m_ref->update(newImage);
   }
   else{
     m_ref->update(newImage);
     genBaseCoordinateMaps(newImage);
     newImage.copyTo(processedImage); 
   }
 
  return processedImage;

}

void WarpingProcessor::genBaseCoordinateMaps(const cuda::GpuMat& image)
{
  //Initialize the arrays
  auto shape = image.size();
  cv::Mat x(shape, CV_32FC1);
  cv::Mat y(shape, CV_32FC1);

  for(int row = 0; row < shape.height; row++)
  {
    for(int col = 0; col < shape.width; col++)
    {
      x.at<float>(row, col) = static_cast<float>(col);
      y.at<float>(row, col) = static_cast<float>(row);
    }
  }
  m_x.upload(x);
  m_y.upload(y);
}

