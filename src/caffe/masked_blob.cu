#include <vector>

#include "caffe/blob.hpp"
#include "caffe/masked_blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>
__global__ void ApplyMask(const int n, const Dtype* in_diff,
    const Dtype* mask, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * mask[index];
    //out_diff[index] = mask[index]?in_diff[index]:(Dtype)0;
  }
}

template <typename Dtype>
void MaskedBlob<Dtype>::_apply_mask_gpu() {
  Dtype* weights_diff = static_cast<Dtype*>(this->diff_->mutable_gpu_data());
  const int count = this->count();
  const Dtype* mask = static_cast<const Dtype*>(this->mask_->gpu_data());
  ApplyMask<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count,weights_diff,mask,weights_diff);
}

template void MaskedBlob<float>::_apply_mask_gpu();
template void MaskedBlob<double>::_apply_mask_gpu();

}