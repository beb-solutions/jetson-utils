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

#include "cudaNV12Surface.h"
#include "cudaYUV.h"	// UD_ENC_FACTOR


// same integer coefficients as the RGB -> I420 conversion (cudaYUV-YV12.cu)
inline __device__ uint8_t nv12_y( const uchar4 px )
{
	return static_cast<uint8_t>(((int)(30 * px.x) + (int)(59 * px.y) + (int)(11 * px.z)) / 100);
}

inline __device__ uchar2 nv12_uv( const uchar4 px )
{
	return make_uchar2(static_cast<uint8_t>(((int)(-17 * px.x) - (int)(33 * px.y) + (int)(50 * px.z) + 12800) / 100),
				    static_cast<uint8_t>(((int)(50 * px.x) - (int)(42 * px.y) - (int)(8 * px.z) + 12800) / 100));
}

// one thread per 2x2 block; surface x coordinates are in bytes
__global__ void RGBAToNV12Surface( uchar4* src, int width, int height, cudaSurfaceObject_t yPlane, cudaSurfaceObject_t uvPlane,
						     uint8_t* data, int data_length, int flip )
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) << 1;

	if( x + 1 >= width || y + 1 >= height )
		return;

	// user data: one bit per block of (1 << UD_ENC_FACTOR)^2 pixels, row by row from the top left
	if( data != NULL && data_length > 0 )
	{
		const uint32_t bit = (y >> UD_ENC_FACTOR) * (width >> UD_ENC_FACTOR) + (x >> UD_ENC_FACTOR);

		if( bit < (uint32_t)(data_length << 3) )
		{
			const uint8_t value = (data[bit >> 3] & (1 << (bit & 7))) ? 255 : 0;

			surf2Dwrite(value, yPlane, x, y);
			surf2Dwrite(value, yPlane, x + 1, y);
			surf2Dwrite(value, yPlane, x, y + 1);
			surf2Dwrite(value, yPlane, x + 1, y + 1);
			surf2Dwrite(make_uchar2(128, 128), uvPlane, x, y >> 1);
			return;
		}
	}

	const int row0 = flip ? (height - y - 1) : y;
	const int row1 = flip ? (height - y - 2) : y + 1;

	const uchar4 p00 = src[row0 * width + x];
	const uchar4 p01 = src[row0 * width + x + 1];
	const uchar4 p10 = src[row1 * width + x];
	const uchar4 p11 = src[row1 * width + x + 1];

	surf2Dwrite(nv12_y(p00), yPlane, x, y);
	surf2Dwrite(nv12_y(p01), yPlane, x + 1, y);
	surf2Dwrite(nv12_y(p10), yPlane, x, y + 1);
	surf2Dwrite(nv12_y(p11), yPlane, x + 1, y + 1);

	// chroma of the bottom right pixel, like the I420 conversion
	surf2Dwrite(nv12_uv(p11), uvPlane, x, y >> 1);
}


// cudaRGBAToNV12Surface
cudaError_t cudaRGBAToNV12Surface( uchar4* input, size_t width, size_t height,
						     cudaSurfaceObject_t yPlane, cudaSurfaceObject_t uvPlane,
						     void* data, size_t data_length, size_t flip )
{
	if( !input || !width || !height || (width & 1) || (height & 1) || !yPlane || !uvPlane )
		return cudaErrorInvalidValue;

	const dim3 block(16, 8);
	const dim3 grid(iDivUp(width >> 1, block.x), iDivUp(height >> 1, block.y));

	RGBAToNV12Surface<<<grid, block>>>(input, width, height, yPlane, uvPlane, (uint8_t*)data, data ? data_length : 0, flip);

	return CUDA(cudaGetLastError());
}
