// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int ksize, const int pad,
    const int stride, const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
__global__ void im2col_tile_gpu_kernel(const int n, const Dtype* data_im,
    const int strideh, const int stridew, 
    const int ksize, 
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out;
    int w_in = w_out;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * strideh + h_in) * stridew + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        *data_col = data_im[i * stridew + j];
        data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, ksize, pad, stride, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void im2col_tile_gpu(const Dtype* data_im, const int channels,
		const int stride_h, const int stride_w,
    const int ksize, Dtype* data_col, 
    const int height_col, const int width_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_tile_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, stride_h, stride_w, ksize, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);

template void im2col_tile_gpu<float>(const float* data_im, const int channels,
		const int stride_h, const int stride_w,
    const int ksize, float* data_col, 
    const int height_col, const int width_col);
template void im2col_tile_gpu<double>(const double* data_im, const int channels,
		const int stride_h, const int stride_w,
    const int ksize, double* data_col, 
    const int height_col, const int width_col);


template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels, const int ksize,
    const int pad, const int stride, const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
    int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
    int coeff_w_col = (1 - stride * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
__global__ void col2im_tile_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels, const int ksize,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize)+ 1;
    int w_col_end = min(w + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize)+ 1;
    int h_col_end = min(h + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
    int coeff_h_col = (1 - ksize * height_col) * width_col;
    int coeff_w_col = (1 - height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im += (c * stride_h + h) * stride_w + w;
    //data_im[index] = val;
    *data_im += val;
  }
}


template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, ksize, pad, stride,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);

template <typename Dtype>
__global__ void copy_stride_gpu_kernel(int n, const Dtype* src_data, 
		const int channels,
		const int height, const int width, Dtype *dst_data, 
		const int stride_h, const int stride_w) {
  CUDA_KERNEL_LOOP(index, n) {
    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);
    
    src_data += (c * height + h) * width + w;
    dst_data += (c * stride_h + h) * stride_w + w;
    *dst_data = *src_data;
  }
}

template <typename Dtype>
void copy_stride_gpu(const Dtype* src_data, 
		const int channels,
		const int height, const int width, Dtype *dst_data, 
		const int stride_h, const int stride_w) {
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  copy_stride_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, src_data, channels, height, width,
      dst_data, stride_h, stride_w);
  CUDA_POST_KERNEL_CHECK;
}

template void copy_stride_gpu<float>(const float* src_data, 
		const int channels,
		const int height, const int width, float *dst_data, 
		const int stride_h, const int stride_w) ;
template void copy_stride_gpu<double>(const double* src_data, 
		const int channels,
		const int height, const int width, double *dst_data, 
		const int stride_h, const int stride_w) ;

template <typename Dtype>
__global__ void copy_stride_gather_gpu_kernel(int n, Dtype* dst_data, 
		const int channels,
		const int height, const int width, const Dtype *src_data, 
		const int stride_h, const int stride_w) {
  CUDA_KERNEL_LOOP(index, n) {
    int w = index % width;
    int h = (index / width) % height;
    int c = index / (width * height);
    
    dst_data += (c * height + h) * width + w;
    src_data += (c * stride_h + h) * stride_w + w;
    *dst_data = *src_data;
  }
}

template <typename Dtype>
void copy_stride_gather_gpu(Dtype* dst_data, 
		const int channels,
		const int height, const int width, const Dtype *src_data, 
		const int stride_h, const int stride_w) {
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  copy_stride_gather_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, dst_data, channels, height, width,
      src_data, stride_h, stride_w);
  CUDA_POST_KERNEL_CHECK;
}

template void copy_stride_gather_gpu<float>(float* dst_data, 
		const int channels,
		const int height, const int width, const float *src_data, 
		const int stride_h, const int stride_w) ;
template void copy_stride_gather_gpu<double>(double* dst_data, 
		const int channels,
		const int height, const int width, const double *src_data, 
		const int stride_h, const int stride_w) ;

template <typename Dtype>
void col2im_tile_gpu(const Dtype* data_col, const int channels,
    const int height_col, const int width_col, const int ksize,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * height * width * channels));
  int height = height_col + ksize - 1;
  int width = width_col + ksize - 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_tile_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, ksize, stride_h,
      stride_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void col2im_tile_gpu<float>(const float* data_col, const int channels,
    const int height_col, const int width_col,
    const int ksize,
    const int stride_h, const int stride_w,
    float* data_im);

template void col2im_tile_gpu<double>(const double* data_col, const int channels,
    const int height_col, const int width_col,
    const int ksize,
    const int stride_h, const int stride_w,
    double* data_im);

}  // namespace caffe
