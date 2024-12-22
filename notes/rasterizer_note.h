// # rasterizer_impl.h, CudaRasterizer:obtain
// obtain [count] * [T] size of memory from [chunck], store the pointer to the memory to [ptr],
// and step [chunk] to the next location. The memory size should be a multiple of [alignment].
//
// (p + alignment - 1) & ~(alignment - 1) is a common way to align memory
template <typename T>
static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

// ===========================================================================================
// # reinterpret_cast, static_cast, const_cast
// Do not use (int*) in modern C++
float num = 42.5f;
int* ptr = reinterpret_cast<int*>(&num);  // Treats the float memory as an integer array
int* int_ptr = static_cast<int*>(ptr);    // Converts float to integer

void modify(const int* ptr) {
    int* non_const_ptr = const_cast<int*>(ptr); // Removes constness
    *non_const_ptr = 42; // Modifies the data
}

// ===========================================================================================
// # rasterizer_impl.cu, CudaRasterizer::Rasterizer::forward
// geometryBuffer is not a tensor.
// Instead, it is a lambda function(geomFunc) that resize empty torch.Tensor and return the pointer to the data.
// The reason is they don't know the size of the buffer beforehand, so they need to dynamically allocate the memory

torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);

// 1. "required" obtains the memory size by stimualting the memory allocation process
// 2. allocate memory with desired size by resize an empty tensor
// 3. allocate the memory to the attributes of GeometryState
size_t chunk_size = required<GeometryState>(P);
char* chunkptr = geometryBuffer(chunk_size);
GeometryState geomState = GeometryState::fromChunk(chunkptr, P);


// ===========================================================================================
// An useful tool that print out cuda error messages when executing function [A] and the flag [debug] being enabled
// e.g. CHECK_CUDA(func, True)
#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

// ===========================================================================================

