#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char* argv[])
{
  cv::Mat image;
  image = cv::imread("../angry_pepe.jpg",1);

  if(!image.data)
  {
    std::cout << "Image had no data!" << std::endl;
    return -1;
  }

  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
cv::imshow("Display Image", image);

cv::waitKey(0);
std::cout << "Thanks for playing!" << std::endl;
return 0;


}


