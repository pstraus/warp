#include <iostream>
#include <opencv2/opencv.hpp>
#include <ExponentialAverage.h>
#include <WarpingProcessor.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>


int main(int argc, char* argv[])
{
  cv::Mat image, processedImage;
  cv::cuda::GpuMat gpu_in, gpu_out;
  std::string filename("testData/base_in_heat_haze.avi");
  std::unique_ptr<WarpingProcessor> processor = nullptr;

  cv::VideoCapture cap("testData/base_in_heat_haze.mp4");
  
  if(!cap.isOpened())
  {
    std::cout << "failed to open file: " << filename << std::endl;
    return -1;
  }

  bool isFirst(true);
  while(cap.isOpened())
  {
    cap >> image;
    gpu_in.upload(image);
    if(image.empty())
    {
      break;
    }
    //Setup processor
    if(isFirst)
    {
      auto refImageGenerator = std::make_unique<ExponentialAverage>(gpu_in, 0.7);
      cv::Ptr<cv::cuda::FarnebackOpticalFlow> flowPtr = cv::cuda::FarnebackOpticalFlow::create( 5, 0.5, false, 25, 10, 5, 1.1, 0);

      processor = std::make_unique<WarpingProcessor>(std::move(refImageGenerator), flowPtr);

      isFirst = false;
    }

    gpu_out = processor->processNewImage(gpu_in);

    gpu_out.download(processedImage);    
    cv::imshow("Processed Video", processedImage);

    cv::imshow("Video Player", image);
    cv::waitKey(25);


    //
  }
  cap.release();
  return 0;
}


