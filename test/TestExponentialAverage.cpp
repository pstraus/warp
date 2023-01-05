//Unit tests ensure CMake is setup properly
#include <gtest/gtest.h>
#include <ExponentialAverage.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

TEST(ExponentialAverageTests, initializes_no_image)
{
  ExponentialAverage test(0.2);
  
  EXPECT_FALSE(test.valid());

}

TEST(ExponentialAverageTests, initializes)
{
  cv::Mat image = cv::imread("testData/angry_pepe.jpg");
  cv::cuda::GpuMat gi(image);

  ExponentialAverage test(gi, 0.2);
  
  EXPECT_TRUE(test.valid());

}

TEST(ExponentialAverageTests, whatsMyAverage)
{
  cv::Mat image1(1,1,CV_32FC1, 100.0f);
  cv::Mat image2(1,1,CV_32FC1, 200.0f);

  cv::cuda::GpuMat gpu_im_1(image1);
  cv::cuda::GpuMat gpu_im_2(image2);

  double alpha = 0.5;

  //Start test
  ExponentialAverage test(gpu_im_1, alpha);
  test.update(gpu_im_2);

  cv::Mat output(1,1,CV_32FC1, 1.0F);
  test.getReferenceImage().download(output);
  EXPECT_DOUBLE_EQ(output.at<float>(0,0), 150.0);
}