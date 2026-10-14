/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
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

#include "gstBufferManager.h"
#include "cudaColorspace.h"
#include "timespec.h"
#include "logging.h"
#include "cudaYUV.h"
#include "cudaWarp.h"


#ifdef ENABLE_NVMM
#include <nvbuf_utils.h>
#include <cuda_egl_interop.h>
#include <NvInfer.h>

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
#include <nvbufsurface.h>   // JetPack 5
#define NVMM_NVBUFSURFACE   // map NVMM buffers with the NvBufSurface API
#endif

#include <pthread.h>
#include <stdlib.h>
#include <mutex>

// Applications may end with exit() while the streaming threads still deliver frames
// (lirecon-vms calls it from its signal handler). The exit handlers of the CUDA runtime
// and EGL tear them down then, and copyNvmm() must not use them anymore: it crashed in
// 3 of 4 shutdowns on Xavier NX. nvmmExitHandler() is registered after CUDA and EGL were
// set up, so it runs before their teardown; it waits for running copies and stops new ones.
static pthread_rwlock_t nvmmExitLock = PTHREAD_RWLOCK_INITIALIZER;
static bool nvmmExiting = false;

static void nvmmExitHandler()
{
	pthread_rwlock_wrlock(&nvmmExitLock);
	nvmmExiting = true;
	pthread_rwlock_unlock(&nvmmExitLock);
}
#endif


// constructor
gstBufferManager::gstBufferManager( videoOptions* options )
{	
	mOptions    = options;
	mFormatYUV  = IMAGE_UNKNOWN;
	mFrameCount = 0;
	mLastTimestamp = 0;
	mNvmmUsed   = false;
	
#ifdef ENABLE_NVMM
	mNvmmStream = NULL;
#endif
	
	mBufferRGB.SetThreaded(false);
	mBufferWarpRGB.SetThreaded(false);
	mBufferUserData.SetThreaded(false);
	mBufferUserDataHelper.SetThreaded(false);
}


// destructor
gstBufferManager::~gstBufferManager()
{
#ifdef ENABLE_NVMM
	if( mNvmmStream != NULL )
		cudaStreamDestroy(mNvmmStream);
#endif
}


#ifdef ENABLE_NVMM
// copyNvmm
bool gstBufferManager::copyNvmm( void* nvmmData )
{
	pthread_rwlock_rdlock(&nvmmExitLock);

	if( nvmmExiting )
	{
		pthread_rwlock_unlock(&nvmmExitLock);
		return false;
	}

	const bool copied = copyNvmmLocked(nvmmData);
	pthread_rwlock_unlock(&nvmmExitLock);

	if( copied )
	{
		static std::once_flag exitHandlerOnce;
		std::call_once(exitHandlerOnce, []() { atexit(nvmmExitHandler); });
	}

	return copied;
}


// copyNvmmLocked
bool gstBufferManager::copyNvmmLocked( void* nvmmData )
{
	// map the NVMM surface as EGLImage
#ifdef NVMM_NVBUFSURFACE
	NvBufSurface* surf = (NvBufSurface*)nvmmData;

	if( !surf || surf->numFilled < 1 || NvBufSurfaceMapEglImage(surf, 0) != 0 )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to map EGLImage from NVMM buffer\n");
		return false;
	}

	EGLImageKHR eglImage = surf->surfaceList[0].mappedAddr.eglImage;
#else
	int nvmmFD = -1;

	if( ExtractFdFromNvBuffer(nvmmData, &nvmmFD) != 0 )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to get FD from NVMM memory\n");
		return false;
	}

	EGLImageKHR eglImage = NvEGLImageFromFd(NULL, nvmmFD);
#endif

	auto unmapImage = [&]()
	{
	#ifdef NVMM_NVBUFSURFACE
		NvBufSurfaceUnMapEglImage(surf, 0);
	#else
		if( eglImage != NULL )
			NvDestroyEGLImage(NULL, eglImage);

		NvReleaseFd(nvmmFD);
	#endif
	};

	if( !eglImage )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to map EGLImage from NVMM buffer\n");
		unmapImage();
		return false;
	}

	// map EGLImage into CUDA
	cudaGraphicsResource* eglResource = NULL;
	cudaEglFrame eglFrame;

	if( CUDA_FAILED(cudaGraphicsEGLRegisterImage(&eglResource, eglImage, cudaGraphicsRegisterFlagsReadOnly)) )
	{
		unmapImage();
		return false;
	}

	auto release = [&]()
	{
		CUDA(cudaGraphicsUnregisterResource(eglResource));
		unmapImage();
	};

	if( CUDA_FAILED(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, eglResource, 0, 0)) )
	{
		release();
		return false;
	}

	if( eglFrame.planeDesc[0].width != mOptions->width || eglFrame.planeDesc[0].height != mOptions->height )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- NVMM EGLImage dimensions mismatch (%ux%u when expected %ux%u)\n", eglFrame.planeDesc[0].width, eglFrame.planeDesc[0].height, mOptions->width, mOptions->height);
		release();
		return false;
	}

	// planes are packed tightly one after another (e.g. NV12: Y @ full res, UV @ half res),
	// which is the same layout as the CPU path delivers
	const uint32_t maxPlanes = 3;
	const uint32_t planeCount = eglFrame.planeCount < maxPlanes ? eglFrame.planeCount : maxPlanes;
	size_t planeWidthBytes[maxPlanes];
	size_t sizeYUV = 0;

	if( mFrameCount == 0 && eglFrame.planeCount != 2 )
		LogWarning(LOG_GSTREAMER "gstBufferManager -- unexpected number of planes in NVMM buffer (%u vs 2 expected)\n", eglFrame.planeCount);

	for( uint32_t n=0; n < planeCount; n++ )
	{
		const cudaChannelFormatDesc& desc = eglFrame.planeDesc[n].channelDesc;
		size_t bytesPerElement = (desc.x + desc.y + desc.z + desc.w) / 8;

		if( bytesPerElement == 0 )
			bytesPerElement = eglFrame.planeDesc[n].numChannels;  // 8-bit channels

		planeWidthBytes[n] = (size_t)eglFrame.planeDesc[n].width * bytesPerElement;
		sizeYUV += planeWidthBytes[n] * eglFrame.planeDesc[n].height;
	}

	if( !mBufferYUV.Alloc(mOptions->numBuffers, sizeYUV, RingBuffer::ZeroCopy) )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to allocate %u image buffers (%zu bytes each)\n", mOptions->numBuffers, sizeYUV);
		release();
		return false;
	}

	uint8_t* nextBuffer = (uint8_t*)mBufferYUV.Peek(RingBuffer::Write);

	if( !nextBuffer )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to retrieve next image ringbuffer for writing\n");
		release();
		return false;
	}

	if( !mNvmmStream && CUDA_FAILED(cudaStreamCreateWithFlags(&mNvmmStream, cudaStreamNonBlocking)) )
	{
		release();
		return false;
	}

	// copy the planes into the ringbuffer on the GPU (block-linear arrays or pitch-linear memory)
	size_t planeOffset = 0;
	bool copied = true;

	for( uint32_t n=0; n < planeCount && copied; n++ )
	{
		const size_t height = eglFrame.planeDesc[n].height;

		if( eglFrame.frameType == cudaEglFrameTypeArray )
			copied = !CUDA_FAILED(cudaMemcpy2DFromArrayAsync(nextBuffer + planeOffset, planeWidthBytes[n], eglFrame.frame.pArray[n], 0, 0, planeWidthBytes[n], height, cudaMemcpyDeviceToDevice, mNvmmStream));
		else
			copied = !CUDA_FAILED(cudaMemcpy2DAsync(nextBuffer + planeOffset, planeWidthBytes[n], eglFrame.frame.pPitch[n].ptr, eglFrame.frame.pPitch[n].pitch, planeWidthBytes[n], height, cudaMemcpyDeviceToDevice, mNvmmStream));

		planeOffset += planeWidthBytes[n] * height;
	}

	// the frame must be complete before the surface is released and the frame is published
	copied = copied && !CUDA_FAILED(cudaStreamSynchronize(mNvmmStream));
	release();

	if( !copied )
		return false;

	mBufferYUV.Next(RingBuffer::Write);
	return true;
}
#endif


// Enqueue
bool gstBufferManager::Enqueue( GstBuffer* gstBuffer, GstCaps* gstCaps )
{
	if( !gstBuffer || !gstCaps )
		return false;

	uint64_t timestamp = apptime_nano();

#if GST_CHECK_VERSION(1,0,0)	
	// map the buffer memory for read access
	GstMapInfo map; 
	
	if( !gst_buffer_map(gstBuffer, &map, GST_MAP_READ) ) 
	{ 
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to map gstreamer buffer memory\n");
		return false;
	}
	
	const void* gstData = map.data;
	const gsize gstSize = map.maxsize; //map.size;

	if( !gstData )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- gst_buffer_map had NULL data pointer...\n");
		return false;
	}

	if( map.maxsize > map.size && mFrameCount == 0 ) 
	{
		LogWarning(LOG_GSTREAMER "gstBufferManager -- map buffer size was less than max size (%zu vs %zu)\n", map.size, map.maxsize);
	}
#else
	// retrieve data pointer
	void* gstData = GST_BUFFER_DATA(gstBuffer);
	const guint gstSize = GST_BUFFER_SIZE(gstBuffer);
	
	if( !gstData )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- gst_buffer had NULL data pointer...\n");
		return false;
	}
#endif
	// on the first frame, print out the recieve caps
	if( mFrameCount == 0 )
		LogVerbose(LOG_GSTREAMER "gstBufferManager recieve caps:  %s\n", gst_caps_to_string(gstCaps));

	// retrieve caps structure
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
	if( !gstCapsStruct )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- gst_caps had NULL structure...\n");
		return false;
	}
	
	// retrieve the width and height of the buffer
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(gstCapsStruct, "width", &width) ||
		!gst_structure_get_int(gstCapsStruct, "height", &height) )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- gst_caps missing width/height...\n");
		return false;
	}
	
	if( width < 1 || height < 1 )
		return false;
	
	mOptions->width = width;
	mOptions->height = height;

	// verify format 
	if( mFrameCount == 0 )
	{
		mFormatYUV = gst_parse_format(gstCapsStruct);
		
		if( mFormatYUV == IMAGE_UNKNOWN )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- stream %s does not have a compatible decoded format\n", mOptions->resource.c_str());
			return false;
		}
		
		LogVerbose(LOG_GSTREAMER "gstBufferManager -- recieved first frame, codec=%s format=%s width=%u height=%u size=%zu\n", videoOptions::CodecToStr(mOptions->codec), imageFormatToStr(mFormatYUV), mOptions->width, mOptions->height, gstSize);
	}

	//LogDebug(LOG_GSTREAMER "gstBufferManager -- recieved %ix%i frame (%zu bytes)\n", width, height, gstSize);
		
#ifdef ENABLE_NVMM
	// check for NVMM buffer
	GstCapsFeatures* gstCapsFeatures = gst_caps_get_features(gstCaps, 0);
	mNvmmUsed = gst_caps_features_contains(gstCapsFeatures, GST_CAPS_FEATURE_MEMORY_NVMM);

	if( mNvmmUsed )
	{
		if( mFrameCount == 0 )
			LogVerbose(LOG_GSTREAMER "gstBufferManager -- recieved NVMM memory\n");

		// copy the frame on the GPU while the GstBuffer is still mapped: the decoder
		// can't reuse its surface before Enqueue() returns, and no NVMM memory is
		// referenced afterwards (no CPU copy, no surface lifetime across threads)
		if( !copyNvmm(map.data) )
		{
			gst_buffer_unmap(gstBuffer, &map);
			return false;
		}
	}
#endif

	// handle CPU path (non-NVMM)
	if( !mNvmmUsed )
	{
		// allocate image ringbuffer
		if( !mBufferYUV.Alloc(mOptions->numBuffers, gstSize, RingBuffer::ZeroCopy) )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to allocate %u image buffers (%zu bytes each)\n", mOptions->numBuffers, gstSize);
			return false;
		}

		// copy to next image ringbuffer
		void* nextBuffer = mBufferYUV.Peek(RingBuffer::Write);

		if( !nextBuffer )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to retrieve next image ringbuffer for writing\n");
			return false;
		}

		memcpy(nextBuffer, gstData, gstSize);
		mBufferYUV.Next(RingBuffer::Write);
	}

		// handle timestamps in either case (CPU or NVMM path)
		size_t timestamp_size = sizeof(uint64_t);

		// allocate timestamp ringbuffer (GPU only if not ZeroCopy)
		if( !mTimestamps.Alloc(mOptions->numBuffers, timestamp_size, RingBuffer::ZeroCopy) )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to allocate %u timestamp buffers (%zu bytes each)\n", mOptions->numBuffers, timestamp_size);
			return false;
		}

		// copy to next timestamp ringbuffer
		void* nextTimestamp = mTimestamps.Peek(RingBuffer::Write);

		if( !nextTimestamp )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to retrieve next timestamp ringbuffer for writing\n");
			return false;
		}

		if (GST_BUFFER_DTS_IS_VALID(gstBuffer) || GST_BUFFER_PTS_IS_VALID(gstBuffer))
		{
			timestamp = GST_BUFFER_DTS_OR_PTS(gstBuffer);
		}

		memcpy(nextTimestamp, (void*)&timestamp, timestamp_size);
		mTimestamps.Next(RingBuffer::Write);

	mWaitEvent.Wake();
	mFrameCount++;
	
#if GST_CHECK_VERSION(1,0,0)
	gst_buffer_unmap(gstBuffer, &map);
#endif
	
	return true;
}


// Dequeue
int gstBufferManager::Dequeue( void** output, imageFormat format, uint64_t timeout )
{
	// wait until a new frame is recieved
	if( !mWaitEvent.Wait(timeout) )
		return 0;

	void* latestYUV = NULL;
	

	// both paths (CPU and NVMM) deliver their frames through the YUV ringbuffer
	latestYUV = mBufferYUV.Next(RingBuffer::ReadLatestOnce);

	if( !latestYUV )
		return 0;

	// handle timestamp (both paths)
	void* pLastTimestamp = NULL;
	pLastTimestamp = mTimestamps.Next(RingBuffer::ReadLatestOnce);

	if( !pLastTimestamp )
	{
		LogWarning(LOG_GSTREAMER "gstBufferManager -- failed to retrieve timestamp buffer (default to 0)\n");
		mLastTimestamp = 0;
	}
	else
	{
		mLastTimestamp = *((uint64_t*)pLastTimestamp);
	}

	// user data: the decoding below reads one row of bit blocks (UD_ENC_FACTOR)
	const bool userDataFits = (mBufferUserData.GetBufferSize() << 3) * (1 << (UD_ENC_FACTOR - 1)) <= (mOptions->width >> 1);

	static bool userDataWarned = false;

	if (mBufferUserData.GetBufferSize() > 0 && !userDataFits && !userDataWarned)
	{
		LogWarning(LOG_GSTREAMER "gstBufferManager -- %zu bytes of user data don't fit into one row of blocks (%u pixels wide), ignoring them\n", mBufferUserData.GetBufferSize(), mOptions->width);
		userDataWarned = true;
	}

	void* gpu_data = NULL; // for user data
	if (mBufferUserData.GetBufferSize() > 0 && userDataFits) {
		size_t num_bits = (mBufferUserData.GetBufferSize() << 3);

		// The encoder writes each bit as a block of (1 << UD_ENC_FACTOR)^2 pixels, row by row
		// from the top left (cudaYUV-YV12.cu). One value is the mean of 2x2 pixels, so the
		// bits below are read from value rows of width/2 values: 1 << (UD_ENC_FACTOR - 1)
		// rows and as many columns per bit. Compute exactly those values (v7 allocated
		// fewer values than it wrote and read values it never computed).
		const size_t value_stride = mOptions->width >> 1;
		const size_t value_rows   = 1 << (UD_ENC_FACTOR - 1);
		const size_t value_cols   = num_bits * value_rows;
		const size_t num_values   = value_stride * value_rows;

		if (!mBufferUserDataHelper.Alloc(1, num_values, RingBuffer::ZeroCopy)) {
			return false;
		}

		uint8_t* gpu_data = (uint8_t*)mBufferUserDataHelper.Next(RingBuffer::Write);
		uint8_t* frame = (uint8_t*)latestYUV;
		const size_t p = mOptions->width;
		
		// get data array
		for(size_t row=0; row < value_rows; row++)
		{
			const uint8_t* line = frame + (row << 1) * p;	// pixel rows 2*row and 2*row+1

			for(size_t col=0; col < value_cols; col++)
			{
				const size_t x = col << 1;

				gpu_data[row * value_stride + col] = (uint8_t)((
					(uint16_t)line[x] + 
					(uint16_t)line[x+1] + 
					(uint16_t)line[p+x] + 
					(uint16_t)line[p+x+1]
				) / 4);
			}
		}
	
		// get bits
		uint8_t* data = (uint8_t*)mBufferUserData.Next(RingBuffer::Write);
		memset(data, 0, mBufferUserData.GetBufferSize());
				
		int byte;
		int bit;
		uint8_t value;

		//CUDA(cudaDeviceSynchronize());

		for(int i=0; i<num_bits; i++) {
			#if UD_ENC_FACTOR == 1
				value = gpu_data[i];	
			#elif UD_ENC_FACTOR == 2
				int j = i << 1;
				int off = mOptions->width >> 1;

				value = ((int)gpu_data[j] + (int)gpu_data[j + 1] + (int)gpu_data[j + off] + (int)gpu_data[j + 1 + off]) >> 2;
			#elif UD_ENC_FACTOR == 3
				int j = i << 2;
				int off = mOptions->width >> 1;

				value = ((int)gpu_data[j] + (int)gpu_data[j + 1] + (int)gpu_data[j + 2] + (int)gpu_data[j + 3] +
					(int)gpu_data[j + off] + (int)gpu_data[j + 1 + off] + (int)gpu_data[j + 2 + off] + (int)gpu_data[j + 3 + off] +
					(int)gpu_data[j + (off << 1)] + (int)gpu_data[j + 1 + (off << 1)] + (int)gpu_data[j + 2 + (off << 1)] + (int)gpu_data[j + 3 + (off << 1)] +
					(int)gpu_data[j + 3* off] + (int)gpu_data[j + 1 + 3* off] + (int)gpu_data[j + 2 + 3* off] + (int)gpu_data[j + 3 + 3* off]) >> 4;
			#endif

			byte = i >> 3;
			bit = i - (byte << 3);

			if (value > 127)
				data[byte] |= (1 << bit); // set bit
			else
				data[byte] &= ~(1 << bit); // reset bit
		}
	}

	// output raw image if conversion format is unknown or NV12
	if ( format == IMAGE_NV12 && mFormatYUV != IMAGE_NV12 )
	{
		// no conversion into NV12: don't hand out e.g. I420 as NV12
		LogError(LOG_GSTREAMER "gstBufferManager -- can't output %s frames as nv12\n", imageFormatToStr(mFormatYUV));
		return -1;
	}

	if ( format == IMAGE_UNKNOWN || format == IMAGE_NV12 )
	{
		*output = latestYUV;
		return 1;
	}

	// allocate ringbuffer for colorspace conversion
	const size_t rgbBufferSize = imageFormatSize(format, mOptions->width, mOptions->height);

	if( !mBufferRGB.Alloc(mOptions->numBuffers, rgbBufferSize, mOptions->zeroCopy ? RingBuffer::ZeroCopy : 0) )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to allocate %u buffers (%zu bytes each)\n", mOptions->numBuffers, rgbBufferSize);
		return -1;
	}

	// perform colorspace conversion
	void* nextRGB = mBufferRGB.Next(RingBuffer::Write);

	if (mBufferUserData.GetBufferSize() == 0 && !mOptions->use_bgra) {
		if( CUDA_FAILED(cudaConvertColor(latestYUV, mFormatYUV, nextRGB, format, mOptions->width, mOptions->height)) )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- unsupported image format (%s)\n", imageFormatToStr(format));
			LogError(LOG_GSTREAMER "                    supported formats are:\n");
			LogError(LOG_GSTREAMER "                       * rgb8\n");		
			LogError(LOG_GSTREAMER "                       * rgba8\n");		
			LogError(LOG_GSTREAMER "                       * rgb32f\n");		
			LogError(LOG_GSTREAMER "                       * rgba32f\n");

			return -1;
		}
	} else {
		if( CUDA_FAILED(cudaConvertColor(latestYUV, mFormatYUV, nextRGB, format, mOptions->width, mOptions->height, 
			NULL, 0, 0, mOptions->use_bgra)) )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- unsupported image format (%s)\n", imageFormatToStr(format));
			LogError(LOG_GSTREAMER "                    supported formats are:\n");
			LogError(LOG_GSTREAMER "                       * rgba8\n");		
			
			return -1;
		}
		/*
		if (mBufferUserDataHelper.GetBufferSize() != 0) {
			uint8_t* data = (uint8_t*)mBufferUserData.Next(RingBuffer::Write);
			memset(data, 0, mBufferUserData.GetBufferSize());

			uint8_t* bcode = (uint8_t*)gpu_data;
			size_t num_bits = (mBufferUserData.GetBufferSize() << 3);
			
							
			int byte;
			int bit;
			uint8_t value;

			CUDA(cudaDeviceSynchronize());

			for(int i=0; i<num_bits; i++) {
				#if UD_ENC_FACTOR == 1
					value = bcode[i];	
				#elif UD_ENC_FACTOR == 2
					int j = i << 1;
					int off = mOptions->width >> 1;

					value = ((int)bcode[j] + (int)bcode[j + 1] + (int)bcode[j + off] + (int)bcode[j + 1 + off]) >> 2;
				#elif UD_ENC_FACTOR == 3
					int j = i << 2;
					int off = mOptions->width >> 1;

					value = ((int)bcode[j] + (int)bcode[j + 1] + (int)bcode[j + 2] + (int)bcode[j + 3] +
						(int)bcode[j + off] + (int)bcode[j + 1 + off] + (int)bcode[j + 2 + off] + (int)bcode[j + 3 + off] +
						(int)bcode[j + (off << 1)] + (int)bcode[j + 1 + (off << 1)] + (int)bcode[j + 2 + (off << 1)] + (int)bcode[j + 3 + (off << 1)] +
						(int)bcode[j + 3* off] + (int)bcode[j + 1 + 3* off] + (int)bcode[j + 2 + 3* off] + (int)bcode[j + 3 + 3* off]) >> 4;
				#endif

				byte = i >> 3;
				bit = i - (byte << 3);

				if (value > 127)
					data[byte] |= (1 << bit); // set bit
				else
					data[byte] &= ~(1 << bit); // reset bit
			}
		}*/
	}

#if 0
	if( !mBufferWarpRGB.Alloc(mOptions->numBuffers, rgbBufferSize, mOptions->zeroCopy ? RingBuffer::ZeroCopy : 0) )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to allocate %u buffers for warp conversion (%zu bytes each)\n", mOptions->numBuffers, rgbBufferSize);
		return -1;
	}
	
	// perform warp
	void* warpRGB = mBufferWarpRGB.Next(RingBuffer::Write);

	// test warp
	float2 focal = make_float2(mOptions->width/2.f, mOptions->height/2.f);
	float2 principal = make_float2(mOptions->width/2.f, mOptions->height/2.f);
	float4 d = make_float4(1.f, 1.0f, 0.f, 0.f);

	if ( CUDA_FAILED(cudaWarpIntrinsic((uchar4*)nextRGB,(uchar4*)warpRGB, mOptions->width, mOptions->height, focal, principal, d))) {
	//if ( CUDA_FAILED(cudaWarpFisheye((uchar4*)nextRGB, (uchar4*)warpRGB, mOptions->width, mOptions->height, 1.0f)) ) {
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to warp");
		return -1;
	}

	*output = warpRGB;
#else
	*output = nextRGB;
#endif
	return 1;
}


// GetUserData
void* gstBufferManager::GetUserData(size_t data_length) {
	if (data_length <= 0) {

		// no data requested: disable and free all buffer
		if (mBufferUserData.GetBufferSize() > 0)
			mBufferUserData.Free();

		return NULL;
	}

	if (data_length != mBufferUserData.GetBufferSize()) {

		// data length has changed: reinitialize the buffer and return no data
		mBufferUserData.Free();
		mBufferUserData.Alloc(1, data_length, RingBuffer::CPU);

		return NULL;
	}

	return mBufferUserData.Next(RingBuffer::ReadLatestOnce);
}