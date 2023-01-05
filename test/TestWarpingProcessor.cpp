//Unit tests ensure CMake is setup properly
#include <gtest/gtest.h>
#include <WarpingProcessor.h>
#include <ExponentialAverage.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>

TEST(ProcessorTests, initializes_nullptr)
{
  //Create null I_Reference Image
  std::unique_ptr<I_ReferenceImage> refImg_null= nullptr;

  //Create using valid values
  std::unique_ptr<I_ReferenceImage> refImg = std::make_unique<ExponentialAverage>(0.1);
  EXPECT_NE(refImg.get(), nullptr);

  cv::Ptr<cv::cuda::FarnebackOpticalFlow> flowPtr_null = nullptr;
  //Create using defaults
  cv::Ptr<cv::cuda::FarnebackOpticalFlow> flowPtr = cv::cuda::FarnebackOpticalFlow::create();
  EXPECT_NE(flowPtr.get(), nullptr);

  EXPECT_NO_THROW(std::make_shared<WarpingProcessor>(std::move(refImg), flowPtr));
}

TEST(ProcessorTests, processImage)
{
  cv::Mat img = cv::imread("testData/angry_pepe.jpg");

  //setup flow & ref image
  std::unique_ptr<I_ReferenceImage> refImg = std::make_unique<ExponentialAverage>(0.1);
  cv::Ptr<cv::cuda::FarnebackOpticalFlow> flowPtr = cv::cuda::FarnebackOpticalFlow::create();

  //Create Processor
  WarpingProcessor test(std::move(refImg), flowPtr);

  // send in second test image.  Output should be the same image
  cv::cuda::GpuMat in(img.size(), img.type());
  in.upload(img);

  cv::cuda::GpuMat out(img.size(), img.type());
  out = test.processNewImage(in);

  cv::Mat outImage(out.size(), img.type());

  //Bring back to CPU memory so we can analyze
  out.download(outImage);

  double diff = cv::norm(img, outImage, cv::NORM_L2);
}