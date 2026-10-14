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

#include "gstNvmmInput.h"
#include "gstUtility.h"
#include "logging.h"

#include <stdlib.h>
#include <string.h>

#ifdef ENABLE_NVMM
#include <NvInfer.h>

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
#define NVMM_INPUT_SUPPORTED   // JetPack 5: NvBufSurface
#endif
#endif


#ifdef NVMM_INPUT_SUPPORTED

#include "cudaNV12Surface.h"

#include <nvbufsurface.h>
#include <cuda_egl_interop.h>
#include <memory>


//-----------------------------------------------------------------------------
// GstMemory that carries the NvBufSurface descriptor. Copies of the memory
// (e.g. deep buffer copies) keep the surface allocation alive as well.
//-----------------------------------------------------------------------------
namespace
{
struct SurfaceMemory
{
	GstMemory memory;
	NvBufSurface descriptor;
	std::shared_ptr<NvBufSurface> owner;
};

struct SurfaceAllocator      { GstAllocator parent; };
struct SurfaceAllocatorClass { GstAllocatorClass parent; };

G_DEFINE_TYPE(SurfaceAllocator, surface_allocator, GST_TYPE_ALLOCATOR)

GstMemory* makeMemory( GstAllocator* allocator, const std::shared_ptr<NvBufSurface>& surface )
{
	SurfaceMemory* memory = new SurfaceMemory();

	memory->descriptor = *surface;
	memory->owner = surface;

	gst_memory_init(&memory->memory, (GstMemoryFlags)0, allocator, NULL, sizeof(NvBufSurface), 0, 0, sizeof(NvBufSurface));
	return &memory->memory;
}

gpointer mapMemory( GstMemory* memory, gsize, GstMapFlags )
{
	return &reinterpret_cast<SurfaceMemory*>(memory)->descriptor;
}

void unmapMemory( GstMemory* ) {}

GstMemory* copyMemory( GstMemory* memory, gssize offset, gssize size )
{
	if( offset != 0 || (size != -1 && size != sizeof(NvBufSurface)) )
		return NULL;

	return makeMemory(memory->allocator, reinterpret_cast<SurfaceMemory*>(memory)->owner);
}

gboolean isSpan( GstMemory*, GstMemory*, gsize* )
{
	return FALSE;
}

void freeMemory( GstAllocator*, GstMemory* memory )
{
	delete reinterpret_cast<SurfaceMemory*>(memory);
}

void surface_allocator_class_init( SurfaceAllocatorClass* klass )
{
	GST_ALLOCATOR_CLASS(klass)->free = freeMemory;
}

void surface_allocator_init( SurfaceAllocator* self )
{
	GstAllocator* allocator = GST_ALLOCATOR(self);

	allocator->mem_type    = "JetsonUtilsNvBufSurface";
	allocator->mem_map     = mapMemory;
	allocator->mem_unmap   = unmapMemory;
	allocator->mem_copy    = copyMemory;
	allocator->mem_share   = copyMemory;
	allocator->mem_is_span = isSpan;
}

GstBuffer* wrapSurface( const std::shared_ptr<NvBufSurface>& surface )
{
	static GstAllocator* allocator = GST_ALLOCATOR(g_object_new(surface_allocator_get_type(), NULL));

	GstBuffer* buffer = gst_buffer_new();
	gst_buffer_append_memory(buffer, makeMemory(allocator, surface));
	return buffer;
}
}


//-----------------------------------------------------------------------------
// one surface with its CUDA view (EGLImage registration and plane surfaces)
//-----------------------------------------------------------------------------
struct gstNvmmInput::Slot
{
	std::shared_ptr<NvBufSurface> surface;

	cudaGraphicsResource* resource = NULL;
	cudaSurfaceObject_t   planes[2] = { 0, 0 };
	bool                  egl = false;

	void releaseView()
	{
		for( int n=0; n < 2; n++ )
		{
			if( planes[n] != 0 )
				cudaDestroySurfaceObject(planes[n]);

			planes[n] = 0;
		}

		if( resource != NULL )
			cudaGraphicsUnregisterResource(resource);

		resource = NULL;

		if( egl )
			NvBufSurfaceUnMapEglImage(surface.get(), 0);

		egl = false;
	}

	bool createView()
	{
		if( NvBufSurfaceMapEglImage(surface.get(), 0) != 0 )
			return false;

		egl = true;

		if( CUDA_FAILED(cudaGraphicsEGLRegisterImage(&resource, surface->surfaceList[0].mappedAddr.eglImage, cudaGraphicsRegisterFlagsNone)) )
		{
			resource = NULL;
			return false;
		}

		cudaEglFrame frame;

		if( CUDA_FAILED(cudaGraphicsResourceGetMappedEglFrame(&frame, resource, 0, 0)) )
			return false;

		if( frame.frameType != cudaEglFrameTypeArray || frame.planeCount != 2 )
		{
			LogError(LOG_GSTREAMER "gstNvmmInput -- unexpected NV12 surface layout (type %i, %u planes)\n", (int)frame.frameType, frame.planeCount);
			return false;
		}

		for( int n=0; n < 2; n++ )
		{
			cudaResourceDesc desc;
			memset(&desc, 0, sizeof(desc));

			desc.resType = cudaResourceTypeArray;
			desc.res.array.array = frame.frame.pArray[n];

			if( CUDA_FAILED(cudaCreateSurfaceObject(&planes[n], &desc)) )
			{
				planes[n] = 0;
				return false;
			}
		}

		return true;
	}

	bool allocate( uint32_t width, uint32_t height )
	{
		releaseView();
		surface.reset();

		NvBufSurfaceCreateParams params;
		memset(&params, 0, sizeof(params));

		params.width       = width;
		params.height      = height;
		params.layout      = NVBUF_LAYOUT_BLOCK_LINEAR;	// pitch-linear input makes the encoder blit
		params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
		params.memType     = NVBUF_MEM_SURFACE_ARRAY;

		NvBufSurface* surf = NULL;

		if( NvBufSurfaceCreate(&surf, 1, &params) != 0 || !surf )
		{
			LogError(LOG_GSTREAMER "gstNvmmInput -- failed to allocate a %ux%u NV12 surface\n", width, height);
			return false;
		}

		surf->numFilled = 1;
		surface = std::shared_ptr<NvBufSurface>(surf, [](NvBufSurface* s) { NvBufSurfaceDestroy(s); });

		return createView();
	}

	~Slot()
	{
		releaseView();
	}
};


// IsEnabled
bool gstNvmmInput::IsEnabled()
{
	const char* env = getenv("JETSON_UTILS_NVMM_ENCODER");
	return !(env != NULL && strcmp(env, "0") == 0);
}


// constructor
gstNvmmInput::gstNvmmInput()
{
	mSlots = new Slot[SlotCount];
	mNext  = 0;
}


// destructor
gstNvmmInput::~gstNvmmInput()
{
	delete[] mSlots;
}


// Render
GstBuffer* gstNvmmInput::Render( void* rgba, uint32_t width, uint32_t height, void* data, size_t data_length, bool flip, bool* busy )
{
	*busy = false;

	if( !rgba || (width & 1) || (height & 1) )
		return NULL;

	for( uint32_t i=0; i < SlotCount; i++ )
	{
		Slot& slot = mSlots[(mNext + i) % SlotCount];

		// still referenced by buffers in the pipeline
		if( slot.surface && slot.surface.use_count() != 1 )
			continue;

		if( !slot.surface || slot.surface->surfaceList[0].width != width || slot.surface->surfaceList[0].height != height )
		{
			if( !slot.allocate(width, height) )
				return NULL;
		}

		if( CUDA_FAILED(cudaRGBAToNV12Surface((uchar4*)rgba, width, height, slot.planes[0], slot.planes[1], data, data_length, flip)) )
			return NULL;

		// the encoder reads the surface as soon as it gets the buffer
		if( CUDA_FAILED(cudaStreamSynchronize(0)) )
			return NULL;

		mNext = (mNext + i + 1) % SlotCount;
		return wrapSurface(slot.surface);
	}

	*busy = true;
	return NULL;
}

#else  // NVMM_INPUT_SUPPORTED

struct gstNvmmInput::Slot {};

bool gstNvmmInput::IsEnabled()					{ return false; }
gstNvmmInput::gstNvmmInput() : mSlots(NULL), mNext(0)	{}
gstNvmmInput::~gstNvmmInput()					{}

GstBuffer* gstNvmmInput::Render( void*, uint32_t, uint32_t, void*, size_t, bool, bool* busy )
{
	*busy = false;
	return NULL;
}

#endif
