from langchain_core.prompts import PromptTemplate


get_restrictions_prompt = PromptTemplate.from_template("""
You are a CUDA kernel optimization expert. Given a CUDA kernel and its tunable parameters, your task is to identify logical restrictions that should be applied to limit the search space to valid and meaningful configurations.

CUDA Kernel:
```
{kernel_string}
```

Tunable Parameters:
```json
{tune_params}
```

Analyze the kernel code and parameters to identify restrictions that ensure:
1. Hardware constraints are respected (e.g., block size limits, shared memory limits)
2. Mathematical relationships between parameters are maintained
3. Invalid configurations are avoided

Provide restrictions as a list of boolean expressions (strings) that must ALL be true for a configuration to be valid. Each restriction should be a single boolean expression using parameter names.

Examples of valid restriction formats:
- "block_size_x*block_size_y<=1024"  # Max threads per block
- "block_size_x==block_size_y*tile_size_y"  # Mathematical relationship
- "block_size_x%32==0"  # Warp alignment


Do not use any functions, for example sizeof
If no restrictions are needed, return an empty list: []
Do not add unnecessary restrictions, remember the main goal is validity not performance.

Example:

Input CUDA kernel:
```
#define WIDTH 4096

__global__ void matmul_kernel(float *C, float *A, float *B) {{

    __shared__ float sA[block_size_y][block_size_x];
    __shared__ float sB[block_size_y][block_size_x];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * block_size_x + tx;
    int y = blockIdx.y * block_size_y + ty;

    float sum = 0.0;
    int k,kb;

    for (k=0; k<WIDTH; k+=block_size_x) {{
        __syncthreads();
        sA[ty][tx] = A[y*WIDTH+k+tx];
        sB[ty][tx] = B[(k+ty)*WIDTH+x];
        __syncthreads();

        for (kb=0; kb<block_size_x; kb++) {{
            sum += sA[ty][kb] * sB[kb][tx];
        }}

    }}

    C[y*WIDTH+x] = sum;
}}
```
Input tunable parameters:
```json
{{"block_size_x":[16, 32, 64], "block_size_y": [1, 2, 4, 8, 16, 32]}}
```

Expected output:
```json
{{"restrictions":["block_size_x==block_size_y"]}}
```

Explanation:
The kernel can only be executed correctly when the area operated on by the thread block as a whole is a square. This means that the number of threads used in the x and y dimensions will have to be equal.

""")