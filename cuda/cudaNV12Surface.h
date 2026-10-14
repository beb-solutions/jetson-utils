/*
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __CUDA_NV12_SURFACE_H
#define __CUDA_NV12_SURFACE_H

#include "cudaUtility.h"


/**
 * Convert RGBA8 into NV12 planes given as CUDA surface objects (e.g. the
 * block-linear arrays of an NvBufSurface). Same coefficients, flip and user
 * data blocks as cudaRGBAToI420(); data must be accessible from the GPU
 * (mapped memory).
 * @ingroup colorspace
 */
cudaError_t cudaRGBAToNV12Surface( uchar4* input, size_t width, size_t height,
						     cudaSurfaceObject_t yPlane, cudaSurfaceObject_t uvPlane,
						     void* data, size_t data_length, size_t flip );

#endif
