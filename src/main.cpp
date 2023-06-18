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
  std::string outputFileName("processed_" + inputFileName ) ;

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
  
  cv::Size imageSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  cv::Size outFrameSize(2*cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  //cv::Size frameSize(cv::CAP_PROP_FRAME_HEIGHT, cv::CAP_PROP_FRAME_WIDTH);

  cv::VideoWriter output(outputFileName
                        //, cv::VideoWriter::fourcc('d','i','v','x')
                        , cap.get(cv::CAP_PROP_FOURCC)
                        , static_cast<uint>(cap.get(cv::CAP_PROP_FPS))
                        , outFrameSize);
  if(!output.isOpened())
  {
    std::cout << "Failed to open video writer! " << std::endl;
    return -1;
  }
  //cv::VideoWriter output(outputFileName, cv::CAP_FFMPEG, cv::VideoWriter::fourcc('m','p','4','v'), 20, frameSize);

  std::unique_ptr<ExponentialAverage> refImageGenerator = nullptr;

  bool isFirst(true);
  cv::Mat image_float;
  cv::Mat img_processed(imageSize, CV_8UC3);
  cv::Mat image_for_writing(outFrameSize, CV_8UC3);
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
    gpu_out = processor->processNewImage(gpu_in);

    gpu_out.download(processedImage);    


    processedImage.convertTo(img_processed, CV_8UC3);
    
    cv::hconcat(image, img_processed, image_for_writing);

    //Write the image
    output << image_for_writing;

  }
  cap.release();
  output.release();
  return 0;
}


