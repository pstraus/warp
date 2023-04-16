#include <iostream>
#include <opencv2/opencv.hpp>
#include <ExponentialAverage.h>
#include <WarpingProcessor.h>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>


int main(int argc, char* argv[])
{
  cv::Mat image, processedImage;
  cv::cuda::GpuMat gpu_in, gpu_out;
  std::string filename("testData/base_in_heat_haze.mp4");
  std::unique_ptr<WarpingProcessor> processor = nullptr;



  cv::VideoCapture cap(filename);
  
  if(!cap.isOpened())
  {
    std::cout << "failed to open file: " << filename << std::endl;
    return -1;
  }

  std::unique_ptr<ExponentialAverage> refImageGenerator = nullptr;

  bool isFirst(true);
  cv::Mat image_float;
  while(cap.isOpened())
  {
    cap >> image;

    image.convertTo(image_float, CV_32F);

    gpu_in.upload(image);
    if(image.empty())
    {
      break;
    }
    //Setup processor
    if(isFirst)
    {
      refImageGenerator = std::make_unique<ExponentialAverage>(gpu_in, 0.7);
      cv::Ptr<cv::cuda::FarnebackOpticalFlow> flowPtr = cv::cuda::FarnebackOpticalFlow::create( );

      processor = std::make_unique<WarpingProcessor>(std::move(refImageGenerator), flowPtr);

      isFirst = false;
    }

    //refImageGenerator->update(gpu_in);
    //gpu_out = refImageGenerator->getReferenceImage();
    gpu_out = processor->processNewImage(gpu_in);

    gpu_out.download(processedImage);    
    cv::imshow("Processed Video", processedImage);

    cv::imshow("Video Player", image);

    cv::Mat diff;
    cv::absdiff(processedImage, image, diff);
    cv::imshow("diff", diff);
    cv::waitKey(25);

  }
  cap.release();
  return 0;
}


