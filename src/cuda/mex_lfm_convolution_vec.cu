/*
 *TODO
 */

/*
 * MEX includes
 */
#include "mex.h"
#include "gpu/mxGPUArray.h"

/*
 * Device code
 */
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

// Texture for the source image.
texture<float, 1, cudaReadModeElementType> texture_bitmap;

// Texture for all kernels. The kernels are stored consecutively.
texture<float, 1, cudaReadModeElementType> texture_kernels;

// Number of threads. We use blocks with dimension [num_threads, 1].
// we need: num_threads + kernel_width < 2 * num_threads,
// so: num_threads > kernel_width
// max possible value is 1024 for GTX Titan X (see specs or output of deviceQuery example app)
// if this is changed then the kernel_width_limit has to be changed, too (see below)
static const int num_threads = 1024;

// Maximum number of repeated kernels in one horizontal line.
static const int num_kernels = 15;

// Maximum horizontal width of a kernel.
// currently, we only implement up to 5 px kernel fetch per thread, so this limit is hard:
// we have to ensure that (kernel_width_limit * num_kernels) / num_threads < 5
// so for num_threads = 1024 the limit is 341, for num_threads = 512 the limit is 170
static const int kernel_width_limit = 341;

void __global__ lfm_convolution_kernel_vec (
    const int bitmap_width, const int bitmap_height,  const int bitmap_ix,
    const int kernel_width,
    const int max_kernel_width,
    const float * const kernels,
    float * const output) {
  // Compute the current pixel coordinates.
  const unsigned int x = (blockIdx.x * num_threads) + threadIdx.x;
  const unsigned int y = blockIdx.y;

  // The convolution kernel will be applied to the rectangle:
  // [bitmap_startx, bitmap_starty, bitmap_endx, bitmap_endy].
  int bitmap_startx = x - kernel_width / 2;
  int bitmap_starty = y - kernel_width / 2;
  int bitmap_endx = x + kernel_width / 2 + 1;
  int bitmap_endy = y + kernel_width / 2 + 1;

  // We store the coordinates of the non-clipped part of the kernel
  // in [kernel_startx, kernel_starty, kernel_endx, kernel_endy].
  int kernel_startx = 0;
  int kernel_starty = 0;
  int kernel_endx = kernel_width;
  int kernel_endy = kernel_width;

  if (bitmap_startx < 0) {
    kernel_startx = -bitmap_startx;
  }
  if (bitmap_starty < 0) {
    kernel_starty = -bitmap_starty;
  }
  if (bitmap_endx > bitmap_width) {
    kernel_endx -= bitmap_endx - bitmap_width;
  }
  if (bitmap_endy > bitmap_height) {
    kernel_endy -= bitmap_endy - bitmap_height;
  }

  // All threads will copy one horizontal line of pixels needed
  // for the convolution to shared memory.
  //
  // This line has coordinates [mstartx, mendx].
  int mstartx = bitmap_startx - threadIdx.x;
  int moffset = 0;
  if (mstartx < 0) {
    moffset = -mstartx;
    mstartx = 0;
  }
  int mendx = bitmap_startx - threadIdx.x + num_threads + kernel_width;
  if (mendx > bitmap_width) {
    mendx = bitmap_width;
  }
  const int mlen = mendx - mstartx;

  // Shared memory for one row needed by the convolution.
  __shared__ float xline[num_threads + kernel_width_limit];

  // Shared memory for pixels from all kernels needed for one row.
  __shared__ float xkernel[num_kernels * kernel_width_limit];

  // Current thread is responsible for copying pixels [bmp_starti, bmp_endi]
  // from the source image to the shared memory.
  const int bmp_starti = (mlen * threadIdx.x) / num_threads;
  const int bmp_endi = (mlen * (threadIdx.x + 1)) / num_threads;

  // Each thread will be responsible for copying one or two pixels, because
  // num_threads + kernel_width < 2 * num_threads.
  const bool bmp_one_step = bmp_endi == bmp_starti + 1;
  const bool bmp_two_step = bmp_endi == bmp_starti + 2;

  // This thread will be responsible for copying pixels
  // [kernel_starti, kernel_endi] from kernels to the shared memory.
  const int kernel_starti = (kernel_width * (threadIdx.x / num_kernels)) /
      (num_threads / num_kernels);
  const int kernel_endi = (kernel_width * ((threadIdx.x / num_kernels) + 1)) /
      (num_threads / num_kernels);

  // Each thread will be responsible for copying 1 to 3 kernel pixels.
  const bool kernel_one_step = kernel_endi == kernel_starti + 1;
  const bool kernel_two_step = kernel_endi == kernel_starti + 2;
  const bool kernel_three_step = kernel_endi == kernel_starti + 3;
  const bool kernel_four_step = kernel_endi == kernel_starti + 4;
  const bool kernel_five_step = kernel_endi == kernel_starti + 5;

  // The current thread will apply this kernel.
  const int current_kernel = (y % num_kernels) * num_kernels + (threadIdx.x % num_kernels); 

  // The kernel in the shared memory.
  const int filling_kernel = threadIdx.x % num_kernels;

  // We will start from this offset in the source image.
  int bitmap_offset = bitmap_ix * bitmap_height * bitmap_width +
                      (bitmap_starty + kernel_starty) * bitmap_width +
                      mstartx +
                      bmp_starti;

  // We will start from this offset in the kernel.
  int kernel_offset = max_kernel_width * max_kernel_width * current_kernel +
                      max_kernel_width * kernel_starty +
                      kernel_starti;

  // The accumulated sum.
  register float value = 0;

  // Compute dot product for every row in the kernel.
  for (int ky = kernel_starty;
       ky < kernel_endy;
       ky++, bitmap_offset += bitmap_width, kernel_offset += max_kernel_width) { 
    // Each thread copies few pixels which it is responsible for from
    // the source image to the shared memory.
    if (bmp_two_step) {
      xline[bmp_starti] = tex1Dfetch(texture_bitmap, bitmap_offset);
      xline[bmp_starti + 1] = tex1Dfetch(texture_bitmap, bitmap_offset + 1);
    } else if (bmp_one_step) {
      xline[bmp_starti] = tex1Dfetch(texture_bitmap, bitmap_offset);
    }

    // Each thread copies few pixels, which it is responsible for from
    // the kernel to the shared memory.
    int o = kernel_starti + filling_kernel * kernel_width;
    if (kernel_five_step) {
      xkernel[o] = tex1Dfetch(texture_kernels, kernel_offset);
      xkernel[o + 1] = tex1Dfetch(texture_kernels, kernel_offset + 1);
      xkernel[o + 2] = tex1Dfetch(texture_kernels, kernel_offset + 2);
      xkernel[o + 3] = tex1Dfetch(texture_kernels, kernel_offset + 3);
      xkernel[o + 4] = tex1Dfetch(texture_kernels, kernel_offset + 4);
    } else if (kernel_four_step) {
      xkernel[o] = tex1Dfetch(texture_kernels, kernel_offset);
      xkernel[o + 1] = tex1Dfetch(texture_kernels, kernel_offset + 1);
      xkernel[o + 2] = tex1Dfetch(texture_kernels, kernel_offset + 2);
      xkernel[o + 3] = tex1Dfetch(texture_kernels, kernel_offset + 3);
    } else if (kernel_three_step) {
      xkernel[o] = tex1Dfetch(texture_kernels, kernel_offset);
      xkernel[o + 1] = tex1Dfetch(texture_kernels, kernel_offset + 1);
      xkernel[o + 2] = tex1Dfetch(texture_kernels, kernel_offset + 2);
    } else if (kernel_two_step) {
      xkernel[o] = tex1Dfetch(texture_kernels, kernel_offset);
      xkernel[o + 1] = tex1Dfetch(texture_kernels, kernel_offset + 1);
    } else if (kernel_one_step) {
      xkernel[o] = tex1Dfetch(texture_kernels, kernel_offset);
    } 

    // We now synchronize all threads. Once the threads are finished, we will
    // have the required row from the source image and one row from all
    // required kernels in the shared memory.
    __syncthreads();

    // Compute the convolution for the pixel [x, y] and the row ky.
    //
    // It is possible that this thread was helping copy data to shared
    // memory, but [x, y] is outside of the image. In that case we are
    // finished.
    if (x < bitmap_width && y < bitmap_height) {
      // The offset in the shared memory for the image.
      int ki = threadIdx.x - moffset;
      if (ki < 0) ki = 0;

      // The offset in the shared memory for the kernel.
      int bi = kernel_width * (x % num_kernels) + kernel_startx;
      const int kend = ki + kernel_endx - kernel_startx;
 
      // Compute the dot product. Unroll to make it faster.
      while (ki + 32 < kend) {
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];

        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];

        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];

        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
      }

      while (ki + 4 < kend) {
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
        value += xline[ki++] * xkernel[bi++];
      }

      while (ki < kend) {
        value += xline[ki++] * xkernel[bi++];
      }
    }

    // Synchronize the threads so that we can move to the next row.
    __syncthreads();
  }

  // Store the accumulated result in the output pixel.
  if (x < bitmap_width && y < bitmap_height) {
    output[x + y * bitmap_width + bitmap_ix * bitmap_width * bitmap_height] = value;
  }
  //if (x < bitmap_width && y < bitmap_height) {
  //  output[x * bitmap_height + y] = value;
  //}
    __syncthreads();
}




/*
 * Host code. First input is 2D single lfm raw image, 
 *            second intput is 4D single LFM PSF for one depth (size: [n_px_x, n_px_y, n_ml_x, n_ml_y])
 *            Return value is a 2D array that is the result of the lfm-convolution.
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const * dev_kernels = 0;
    mxGPUArray const * dev_bitmap = 0;
    float const * dev_kernels_p = 0;
    float const * dev_bitmap_p = 0;
    mxGPUArray * dev_output = 0;
    float * dev_output_p = 0;
    int bitmap_width;
    int bitmap_height;
    int bitmap_n;
    // kernel_width is currently not really used, but it could be set to a value smaller than max_kernel_width 
    // to skip unnecessary mutliplications if nonzero kernel pixels occupy an area smaller than max_kernel_width^2
    // in that case we would have to add an input arg to mexFunction that specifies the current effective width
    int kernel_width;
    // max_kernel_width is the stride at which kernels are stored in dev_kernels_p
    int max_kernel_width;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to mex_lfm_convolution_vec()";


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the first and second inputs are not GPU arrays. */
    //if ((nrhs!=2) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1]))) {
    if (nrhs!=2) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    dev_bitmap = mxGPUCreateFromMxArray(prhs[0]);
    dev_kernels = mxGPUCreateFromMxArray(prhs[1]);
    
    mwSize * dev_bitmap_dims_p = (mwSize *) mxGPUGetDimensions(dev_bitmap);
    if (mxGPUGetNumberOfDimensions(dev_bitmap) <= 1 || mxGPUGetNumberOfDimensions(dev_bitmap) > 3) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    if (mxGPUGetNumberOfDimensions(dev_bitmap) >= 2) {
        bitmap_width = dev_bitmap_dims_p[0];
        bitmap_height = dev_bitmap_dims_p[1];
        bitmap_n = 1;
    }
    if (mxGPUGetNumberOfDimensions(dev_bitmap) == 3) {
        bitmap_n = dev_bitmap_dims_p[2];
    }
    //mwSize tmp;
    //tmp = dev_bitmap_dims_p[0];
    //dev_bitmap_dims_p[0] = dev_bitmap_dims_p[1];
    //dev_bitmap_dims_p[1] = tmp;
    //bitmap_width = dev_bitmap_dims_p[0];
    //bitmap_height = dev_bitmap_dims_p[1];
    //bitmap_n = dev_bitmap_dims_p[2];
    mexPrintf("\n---\nbitmap_width: %i  bitmap_height: %i  bitmap_n: %i\n", bitmap_width, bitmap_height, bitmap_n);
    if (bitmap_n < 1) {
        mexPrintf("bitmap_n=%i has to be at least 1\n", bitmap_n);
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    if (4 != mxGPUGetNumberOfDimensions(dev_kernels)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    const mwSize * dev_kernels_dims_p = mxGPUGetDimensions(dev_kernels);
    kernel_width = dev_kernels_dims_p[0];
    max_kernel_width = dev_kernels_dims_p[0];
    mexPrintf("max_kernel_width: %i\n", max_kernel_width);
    if (max_kernel_width > kernel_width_limit) {
        mexPrintf("kernel_width=%i exceeds kernel_width_limit=%i\n", max_kernel_width, kernel_width_limit);
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    //TODO: check if kernels are square

    /*
     * Verify that both inputs really are single arrays before extracting the pointers.
     */
    if ((mxGPUGetClassID(dev_bitmap) != mxSINGLE_CLASS) || (mxGPUGetClassID(dev_kernels) != mxSINGLE_CLASS)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    cudaDeviceSynchronize();
    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    dev_bitmap_p = (float const *)(mxGPUGetDataReadOnly(dev_bitmap));
    dev_kernels_p = (float const *)(mxGPUGetDataReadOnly(dev_kernels));

    /* Create a GPUArray to hold the result and get its underlying pointer. TN removed: mxGPUGetDimensions(dev_bitmap)  dev_bitmap_dims_p */
    dev_output = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(dev_bitmap),
                            dev_bitmap_dims_p,
                            mxGPUGetClassID(dev_bitmap),
                            mxGPUGetComplexity(dev_bitmap),
                            MX_GPU_INITIALIZE_VALUES); //MX_GPU_DO_NOT_INITIALIZE
    dev_output_p = (float *)(mxGPUGetData(dev_output));
    cudaDeviceSynchronize();

    size_t texture_bitmap_offset = 0;
    cudaError_t err;
    err = cudaBindTexture(NULL, texture_bitmap, dev_bitmap_p, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat), bitmap_width * bitmap_height * bitmap_n * sizeof(float));
    if (err != cudaSuccess) {
        printf("texture_bitmap_offset:%i  err:%i\n", texture_bitmap_offset, err);
        mexErrMsgIdAndTxt(cudaGetErrorString(err), "Error in cudaBindTexture() for bitmap");
    }
    size_t texture_kernels_offset = 0;
    err = cudaBindTexture(NULL, texture_kernels, dev_kernels_p, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat), max_kernel_width * max_kernel_width * num_kernels * num_kernels * sizeof(float));
    if (err != cudaSuccess) {
        printf("texture_bitmap_offset:%i  err:%i\n", texture_kernels_offset, err);
        mexErrMsgIdAndTxt(cudaGetErrorString(err), "Error in cudaBindTexture() for kernel");
    }
    cudaDeviceSynchronize();
    
    /*
     * Call the kernel using the CUDA runtime API
     */
    dim3 block(bitmap_width / num_threads + 1, bitmap_height);
    dim3 threadsPerBlock(num_threads, 1);
    printf("px_per_thread: %i  num_threads: %i\n", bitmap_width / num_threads + 1, num_threads);
    printf("kernel_width: %i  max_kernel_width: %i  kernel_width_limit: %i dev_output_p: %u\n", kernel_width, max_kernel_width, kernel_width_limit, dev_output_p);
    
    const int num_streams = 16;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < bitmap_n; i++) {
        lfm_convolution_kernel_vec<<<block, threadsPerBlock, 0, streams[i % num_streams]>>>(
      bitmap_width, bitmap_height, i, kernel_width, max_kernel_width, dev_kernels_p, dev_output_p);
    }
    cudaDeviceSynchronize();

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(dev_output);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    cudaUnbindTexture(texture_bitmap);
    cudaUnbindTexture(texture_kernels);
    mxGPUDestroyGPUArray(dev_bitmap);
    mxGPUDestroyGPUArray(dev_kernels);
    mxGPUDestroyGPUArray(dev_output);
    cudaDeviceSynchronize();
}
