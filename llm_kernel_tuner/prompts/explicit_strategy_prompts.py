from langchain_core.prompts import PromptTemplate

system_prompt = "You are a helpful assistant that rewrites CUDA code."

step_evaluation_prompt = PromptTemplate.from_template("""
You are an expert in CUDA programming and kernel optimization. Your task is to analyze a given CUDA kernel and determine whether a proposed optimization technique makes sense to implement.
First, examine the following CUDA kernel code:
```CUDA
{kernel_string}
```
Now, consider the following proposed optimization technique prompt:
\"\"\"
{optimization_technique}
\"\"\"
Based on your expertise, please answer the following question:
Does the proposed optimization technique make sense to implement and test in the given CUDA kernel?
""")

process2elem_x = PromptTemplate.from_template("""
Rewrite the following CUDA code to process 2 elements in the x dimension:
{kernel_string}
Do not change the argument list of the kernel.
""")

tunable_nr_elem_x = PromptTemplate.from_template("""
Please rewrite the following CUDA code by introducing a for loop to allow the number of elements processed by each thread to vary in the x dimension. Please call this constant 'work_per_thread_x' in lower case.
{kernel_string}
Do not change the argument list of the kernel.
""")

tile_stride_x = PromptTemplate.from_template("""
Rewrite the following CUDA code by introducing a new preprocessor constant 'work_stride_x'. This constant will determine the thread work pattern: working on 'work_per_thread_x' consecutive elements when 'work_stride_x' is 0, or 'work_per_thread_x' elements spaced by blockDim.x when 'work_stride_x' is 1. When work_stride_x is 0, multiply threadIdx.x by work_per_thread_x.
{kernel_string}
Do not change the argument list of the kernel.
""")

process2elem_y = PromptTemplate.from_template("""
Rewrite the following CUDA code to process 2 elements in the y dimension:
{kernel_string}
Do not change the argument list of the kernel.
""")

tunable_nr_elem_y = PromptTemplate.from_template("""
Please rewrite the following CUDA code by introducing a for loop to allow the number of elements processed by each thread to vary in the x dimension. Please call this constant 'work_per_thread_x' in lower case.
{kernel_string}
Do not change the argument list of the kernel.
""")

tile_stride_y = PromptTemplate.from_template("""
Rewrite the following CUDA code by introducing a new preprocessor constant 'work_stride_x'. This constant will determine the thread work pattern: working on 'work_per_thread_x' consecutive elements when 'work_stride_x' is 0, or 'work_per_thread_x' elements spaced by blockDim.x when 'work_stride_x' is 1. When work_stride_x is 0, multiply threadIdx.x by work_per_thread_x.
{kernel_string}
Do not change the argument list of the kernel.
""")

shared_memory_input = PromptTemplate.from_template("""
Revise the CUDA code to include a C preprocessor constant 'use_shared_mem', allowing developers to opt for input data caching in shared memory.
If 'use_shared_mem' is 0, avoid shared memory caching and declaration.
Conversely, if it's 1, declare static shared memory and load all input data there before computation.
Thread block dimensions are predefined via 'block_size_x' and 'block_size_y' constants.
""")

shared_memory_output = PromptTemplate.from_template("""
Modify the CUDA code to introduce a 'output_shared_mem' preprocessor constant, enabling developers to dictate if output data should be collected in shared memory before global memory storage.
If 'output_shared_mem' is 0, bypass shared memory storage and declaration.
If it's 1, declare static shared memory and initially store all output data there before moving to global memory.
Thread block dimensions are predefined via 'block_size_x' and 'block_size_y' constants.
""")

loop_unrolling = PromptTemplate.from_template("""
Rewrite the following CUDA code to unroll the inner-most loop. The unroll factor should be a tunable parameter called 'loop_unroll_factor'.
{kernel_string}
Do not change the argument list of the kernel.
""")

instruction_parallelism = PromptTemplate.from_template("""
Rewrite the following CUDA code to increase instruction-level parallelism by using __syncthreads() to overlap memory operations with computation.
{kernel_string}
Do not change the argument list of the kernel.
""")

multi_dimensional_tiling_x = PromptTemplate.from_template("""
Rewrite the following CUDA code to introduce 2D tiling. This will be a two-step process. First, add tiling in the x-dimension. The tile size in the x-dimension should be a tunable parameter called 'tile_size_x'.
{kernel_string}
Do not change the argument list of the kernel.
""")

multi_dimensional_tiling_y = PromptTemplate.from_template("""
Rewrite the following CUDA code to add tiling in the y-dimension. The tile size in the y-dimension should be a tunable parameter called 'tile_size_y'. This step should only be performed after tiling in the x-dimension has been applied.
{kernel_string}
Do not change the argument list of the kernel.
""")

memory_prefetching = PromptTemplate.from_template("""
Rewrite the following CUDA code to prefetch data into the L1 cache. This should be done using the `__builtin_prefetch` intrinsic. This step should only be performed after shared memory has been enabled.
{kernel_string}
Do not change the argument list of the kernel.
""")

block_size_x = PromptTemplate.from_template("""
Rewrite the following CUDA code to make the block size in the x-dimension a tunable parameter called 'block_size_x'.
{kernel_string}
Do not change the argument list of the kernel.
""")

block_size_y = PromptTemplate.from_template("""
Rewrite the following CUDA code to make the block size in the y-dimension a tunable parameter called 'block_size_y'.
{kernel_string}
Do not change the argument list of the kernel.
""")


