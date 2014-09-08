// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  if(NTILE_WIDTH_ * NTILE_HEIGHT_ <= 1){
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    Dtype* col_data = col_buffer_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    int weight_offset = M_ * K_;
    int col_offset = K_ * N_;
    int top_offset = M_ * N_;
    for (int n = 0; n < num_; ++n) {
      // First, im2col
      im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
                        width_, kernel_size_, pad_, stride_, col_data);
      // Second, innerproduct with groups
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
          (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
          (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
      }
      // third, add bias
      if (bias_term_) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
            reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
            (Dtype)1., top_data + (*top)[i]->offset(n));
      }
    }
  }
  }else{
   const Dtype* bottom_data = bottom[0]->gpu_data();
   Dtype* top_data = (*top)[0]->mutable_gpu_data();
   Dtype* col_data = col_buffer_.mutable_gpu_data();
    CHECK_EQ(stride_, 1);
    CHECK_EQ(pad_, 0);
    CHECK_EQ(group_, 1);
    CHECK_EQ(col_buffer_.height(), TILE_HEIGHT_);
    Dtype *out_buffer = out_buffer_.mutable_gpu_data();
    for (int n = 0; n < num_; ++n) {
      for(int ny = 0; ny < NTILE_HEIGHT_; ny++){
        for(int nx = 0; nx < NTILE_WIDTH_; nx++){
          int idx = ny * NTILE_WIDTH_ + nx;
          const Dtype* weight = this->blobs_[idx]->gpu_data();
          const Dtype * img = bottom_data + bottom[0]->offset(n, 0,
                TILE_HEIGHT_ * ny, TILE_WIDTH_ * nx);
          im2col_tile_gpu(img,   channels_, height_,
              width_, kernel_size_, col_data,
              TILE_HEIGHT_, TILE_WIDTH_);
	  //dump(&col_buffer_);
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., weight, col_data, (Dtype)0., out_buffer);
	  //dump(&out_buffer_);
          if (bias_term_) {
            const Dtype *bias_ptr = this->blobs_[idx + NTILE_WIDTH_ *
		    NTILE_HEIGHT_]->gpu_data();
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                N_, 1, (Dtype)1., bias_ptr,
                reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
                (Dtype)1., out_buffer);
          }
	  /* copy back */

	  int height_out = height_ - kernel_size_ + 1;
	  int width_out = width_ - kernel_size_ + 1;
	  copy_stride_gpu(out_buffer, num_output_, TILE_HEIGHT_, TILE_WIDTH_,
		top_data + (*top)[0]->offset(n, 0, TILE_HEIGHT_*ny,
			TILE_WIDTH_*nx), height_out, width_out);

        }
      }
    }/* n */
  }

  return Dtype(0.);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  int ntiles = NTILE_WIDTH_ * NTILE_HEIGHT_;
  if(ntiles <= 1){
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  Dtype* bias_diff = NULL;
  if (bias_term_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = M_ * K_;
  const int col_offset = K_ * N_;
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = (*bottom)[i]->gpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (bias_term_) {
      for (int n = 0; n < num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
            1., top_diff + top[0]->offset(n),
            static_cast<const Dtype*>(bias_multiplier_->gpu_data()),
            1., bias_diff);
      }
    }
    for (int n = 0; n < num_; ++n) {
      // since we saved memory in the forward pass by not storing all col data,
      // we will need to recompute them.
      im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                        width_, kernel_size_, pad_, stride_, col_data);
      // gradient w.r.t. weight. Note that we will accumulate diffs.
      for (int g = 0; g < group_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
          (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
          col_data + col_offset * g, (Dtype)1.,
          weight_diff + weight_offset * g);
      }
      // gradient w.r.t. bottom data, if necessary
      if (propagate_down[i]) {
        for (int g = 0; g < group_; ++g) {
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
            (Dtype)1., weight + weight_offset * g,
            top_diff + top[i]->offset(n) + top_offset * g,
            (Dtype)0., col_diff + col_offset * g);
        }
        // col2im back to the data
        col2im_gpu(col_diff, channels_, height_, width_, kernel_size_, pad_,
            stride_, bottom_diff + (*bottom)[i]->offset(n));
      }
    }
  }
  }else{
   Dtype* col_data = col_buffer_.mutable_gpu_data();
   Dtype* col_diff = col_buffer_.mutable_gpu_diff();
   // bias gradient if necessary
   Dtype* bias_diff = NULL;

    CHECK_EQ(group_, 1);
    CHECK_EQ(top.size(), 1);
    Dtype *out_buffer = out_buffer_.mutable_gpu_data();
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    for(int i = 0; i < ntiles; i++){
	    if (bias_term_) {
		    bias_diff = this->blobs_[ntiles + i]->mutable_gpu_diff();
		    CUDA_CHECK(cudaMemset(bias_diff, 0,
					    sizeof(Dtype) * this->blobs_[ntiles+i]->count()));
	    }

	    CUDA_CHECK(cudaMemset(this->blobs_[i]->mutable_gpu_diff(), 0,
				    sizeof(Dtype) * this->blobs_[i]->count()));
    }
    //XXX overlap region ??
    CUDA_CHECK(cudaMemset(bottom_diff, 0,
		    sizeof(Dtype) * (*bottom)[0]->count()));

    for (int n = 0; n < num_; ++n) {
	    for(int ny = 0; ny < NTILE_HEIGHT_; ny++){
		    for(int nx = 0; nx < NTILE_WIDTH_; nx++){
			    int idx = ny * NTILE_WIDTH_ + nx;
			    Dtype* weight_diff =
				    this->blobs_[idx]->mutable_gpu_diff();
			    const Dtype * img = bottom_data + (*bottom)[0]->offset(n, 0,
					    TILE_HEIGHT_ * ny, TILE_WIDTH_ * nx);
			    im2col_tile_gpu(img,   channels_, height_,
					    width_, kernel_size_, col_data,
					    TILE_HEIGHT_, TILE_WIDTH_);

			    int height_out = height_ - kernel_size_ + 1;
			    int width_out = width_ - kernel_size_ + 1;

			    const Dtype* top_diff = top[0]->gpu_diff();
			    copy_stride_gather_gpu(out_buffer, num_output_, TILE_HEIGHT_, TILE_WIDTH_,
					    top_diff + top[0]->offset(n, 0, TILE_HEIGHT_*ny,
						    TILE_WIDTH_*nx), height_out, width_out);

			    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
					    (Dtype)1., out_buffer,
					    col_data, (Dtype)1.,
					    weight_diff);
			    if(bias_term_) {
				    bias_diff = this->blobs_[ntiles + idx]->mutable_gpu_diff();
				    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
						    1., out_buffer,
						    reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
						    1., bias_diff);
			    }
			    if(propagate_down[0]){
				    const Dtype* weight = this->blobs_[idx]->gpu_data();
				    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
						    (Dtype)1., weight,
						    out_buffer,
						    (Dtype)0., col_diff);
				    // col2im back to the data
				    col2im_tile_gpu(col_diff, channels_,
						    TILE_HEIGHT_, TILE_WIDTH_,
						    kernel_size_, height_, width_,
						    bottom_diff +
						    (*bottom)[0]->offset(n,0, TILE_HEIGHT_*ny, TILE_WIDTH_*nx));
			    }
		    }
	    }
    }

  }
}


INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
