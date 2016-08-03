#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/masked_blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
//
namespace caffe {
//template <typename Dtype>
//MaskedBlob<Dtype>::MaskedBlob(const int num, const int channels, const int height,
//    const int width) {
//  Blob<Dtype>(num, channels, height, width);
//}
//
//template <typename Dtype>
//MaskedBlob<Dtype>::MaskedBlob(const vector<int>& shape) {
//  this->Blob<Dtype>(shape);
//}

template <typename Dtype>
void MaskedBlob<Dtype>::setMask(const shared_ptr<Blob<Dtype> > mask){
  this->mask_=mask;
}


// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void MaskedBlob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void MaskedBlob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void MaskedBlob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (this->data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    {
      // apply mask on CPU
      //LOG(INFO) << "Masked Update (CPU).";
      Dtype * weights_diff = static_cast<Dtype*>(this->diff_->mutable_cpu_data());
      const int count = this->count_;
      const Dtype* mask = static_cast<const Dtype*>(this->mask_->cpu_data());
      for (int i = 0; i < count; ++i) {
         weights_diff[i] = weights_diff[i] * mask[i];
      }
      // perform computation on CPU
      caffe_axpy<Dtype>(this->count_, Dtype(-1),
          static_cast<const Dtype*>(this->diff_->cpu_data()),
          static_cast<Dtype*>(this->data_->mutable_cpu_data()));
    }
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // apply mask on GPU
    //LOG(INFO) << "Masked Update (GPU).";
    this->_apply_mask_gpu();
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(this->count_, Dtype(-1),
        static_cast<const Dtype*>(this->diff_->gpu_data()),
        static_cast<Dtype*>(this->data_->mutable_gpu_data()));
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}
INSTANTIATE_CLASS(MaskedBlob);
}
