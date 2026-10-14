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

#ifndef __GSTREAMER_NVMM_INPUT_H__
#define __GSTREAMER_NVMM_INPUT_H__

#include <gst/gst.h>
#include <stdint.h>
#include <stddef.h>


/**
 * Encoder input in NVMM memory: RGBA8 frames are converted with CUDA straight
 * into NV12 NvBufSurfaces, which are handed to the V4L2 encoders without
 * nvvidconv. A surface is written again only after GStreamer released every
 * buffer that wraps it.
 *
 * Set the environment variable JETSON_UTILS_NVMM_ENCODER=0 to disable it.
 * @ingroup codec
 */
class gstNvmmInput
{
public:
	/**
	 * Is NVMM encoder input compiled in and not disabled?
	 */
	static bool IsEnabled();

	gstNvmmInput();
	~gstNvmmInput();

	/**
	 * Convert an RGBA8 CUDA image into a free surface and wrap it into a GstBuffer.
	 * @param data user data blocks (GPU accessible), see cudaRGBAToNV12Surface()
	 * @param busy set to true when no surface is free (skip the frame)
	 * @returns the buffer, or NULL when busy or on failure
	 */
	GstBuffer* Render( void* rgba, uint32_t width, uint32_t height, void* data, size_t data_length, bool flip, bool* busy );

private:
	struct Slot;

	static const uint32_t SlotCount = 4;

	Slot*    mSlots;
	uint32_t mNext;
};

#endif
