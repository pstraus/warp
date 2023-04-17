#include <iostream>
#include <opencv2/opencv.hpp>
#include <ExponentialAverage.h>
#include <WarpingProcessor.h>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>


int main(int argc, char* argv[])
{
  std::string inputFileName(argv[1]);
  std::string outputFileName("processed_" + inputFileName) ;

  cv::Mat image, processedImage;
  cv::cuda::GpuMat gpu_in, gpu_out;
  //std::string filename("testData/base_in_heat_haze.mp4");
  std::unique_ptr<WarpingProcessor> processor = nullptr;



  cv::VideoCapture cap(inputFileName);
  
  if(!cap.isOpened())
  {
    std::cout << "failed to open file: " << inputFileName << std::endl;
    return -1;
  }
  cv::Size frameSize(cv::CAP_PROP_FRAME_WIDTH, cv::CAP_PROP_FRAME_HEIGHT);

  cv::VideoWriter output(outputFileName, cv::VideoWriter::fourcc('m','p','4','v'), cap.get(cv::CAP_PROP_FPS), frameSize);

  std::unique_ptr<ExponentialAverage> refImageGenerator = nullptr;

  bool isFirst(true);
  cv::Mat image_float;
  while(cap.isOpened())
  {
    cap >> image;

    image.convertTo(image_float, CV_32F);

    if(image.empty())
    {
      break;
    }


    gpu_in.upload(image);
    //Setup processor
    if(isFirst)
    {
      refImageGenerator = std::make_unique<ExponentialAverage>(gpu_in, 0.7);

      //There is also a native cuda optical flow we can leverage
      cv::Ptr<cv::cuda::FarnebackOpticalFlow> flowPtr = cv::cuda::FarnebackOpticalFlow::create( );

      processor = std::make_unique<WarpingProcessor>(std::move(refImageGenerator), flowPtr);

      isFirst = false;
    }

    //refImageGenerator->update(gpu_in);
    //gpu_out = refImageGenerator->getReferenceImage();
    gpu_out = processor->processNewImage(gpu_in);

    gpu_out.download(processedImage);    
    output << processedImage ;
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


