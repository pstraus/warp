//Unit tests ensure CMake is setup properly
#include <gtest/gtest.h>
#include <warp/ExponentialAverage.h>


TEST(ExponentialAverageTests, readImage)
{
  cv::Mat image;
  image = cv::imread("../angry_pepe.jpg",1);

  if(!image.data)
  {
    std::cout << "Image had no data!" << std::endl;
  }

  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
cv::imshow("Display Image", image);

cv::waitKey(1);
std::cout << "Thanks for playing!" << std::endl;

}