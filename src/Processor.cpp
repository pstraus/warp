// paul.straus@gmail.com
// all rights reserved

// file Processor.cpp

//Includes
#include <Processor.h>
#include <memory>


    //Constructor consumes unique_ptr of type I_ReferenceImage
    Processor::Processor(std::unique_ptr<I_ReferenceImage>&& refImage)
    {

    }


    //actual functions
    cv::cuda::GpuMat processNewImage(cv::cuda::GpuMat& newImage) override;
