#ifndef CAFFE_MASKED_BLOB_HPP_
#define CAFFE_MASKED_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>
class MaskedBlob: public Blob<Dtype> {
  public:
    void Update(); 
    void setMask(const shared_ptr<Blob<Dtype> >);
      
    MaskedBlob(): Blob<Dtype>() {}
    /// @brief Deprecated; use <code>MaskedBlob(const vector<int>& shape)</code>.
    explicit MaskedBlob(const int num, const int channels, const int height,
        const int width): Blob<Dtype>(num,channels,height,width) {}
    explicit MaskedBlob(const vector<int>& shape): Blob<Dtype>(shape) {}
  protected:
    shared_ptr<Blob<Dtype> > mask_;
  private:
    void _apply_mask_gpu();
};

}

#endif 