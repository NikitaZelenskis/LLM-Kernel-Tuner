from langchain_core.prompts import PromptTemplate


kernel_description_prompt = PromptTemplate.from_template("""
Give a brief description of what the goal of this kernel is. 
Do not describe how the memory is accessed, or what each thread is doing.
Your answer will be used as a description to a next prompt to transform the kernel.
Here is the kernel:
```CUDA\n{kernel_string}\n```""")


problem_size_prompt = """Task:
User will prvide you with a kernel, analyze it to determine the appropriate `problem_size`.
The `problem_size` represents the size of the data domain from which the grid dimensions of the kernel are computed, typically the number of elements the kernel processes.
Analyze the kernel code to identify the size of the data it processes.

Examples:
Vector Addition Kernel:
    Example input:
    ```CUDA
    __global__ void vectorAdd(float *A, float *B, float *C, int n) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < n) {
            C[idx] = A[idx] + B[idx];
        }
    }```
    Reasoning:
        The kernel only needs N threads to process N elements.
    Example output:
    {{"problem_size": ["n"]}}
    ```
                                                   
Matrix Multiplication Kernel:
    Example input:
    ```CUDA
__global__ void matrixMultiply(float *A, int num_dimentions, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_dimentions && col < num_dimentions) {
        float value = 0.0f;
        for (int k = 0; k < num_dimentions; k++) {
            value += A[row * num_dimentions + k] * B[k * num_dimentions + col];
        }
        C[row * num_dimentions + col] = value;
    }
}```
    Reasoning:
        The kernel processes a 2D matrix of size N x N.
    Example output:
    {{"problem_size": ["num_dimentions", "num_dimentions"]}}

Reduction Kernel:
    Example input:
    ```CUDA
__global__ void reduceSum(int num_elem, float *in, float *out) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0;
    if (idx < num_elem) sum += in[idx];
    if (idx + blockDim.x < num_elem) sum += in[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>=1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}```
                                                   
    Reasoning:
        The kernel processes the number of elements in the input array.
    Example output:
    {{"problem_size": ["num_elem"]}}
"""


extract_output_var_prompt = """
Task:
User will provide you with a CUDA kernel, analyze it and determine which variable(s) are the ouput variables. 

Examples:
Vector Addition Kernel:
    Example input:
    ```CUDA
    __global__ void vectorAdd(float *A, float *B, float *C, int n) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < n) {
            C[idx] = A[idx] + B[idx];
        }
    }```
    Reasoning:
        C variable is used as output 
    Example output:
    {{"output_variables": ["C"]}}
    
Square and Cube Kernel:
    Example input:
    ```
    __global__ void computeSquareAndCube(const float *input, float *squares, float *cubes, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            squares[idx] = input[idx] * input[idx];
            cubes[idx] = input[idx] * input[idx] * input[idx];
        }
    }```
    Reasoning:
        `computeSquareAndCube` has 2 outputs, one is `squares` and the other is `cubes`
    Example output:
    {{"output_variables": ["squares", "cubes"]}}
"""
