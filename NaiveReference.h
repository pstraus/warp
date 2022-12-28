// paul.straus@gmail.com
// all rights reserved


// file I_ReferenceImage.h

//Includes
#include <warp/I_ReferenceImage.h>

/// \brief abstract class describing what a reference image must do / be updated
class NaiveReference : public I_ReferenceImage
{
  public:
    cv::cuda::GpuMat& getReferenceImage() const override;
    void update(const cv::Mat& newImage) override;
    bool valid() const override;
}