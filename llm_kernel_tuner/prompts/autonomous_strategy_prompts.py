from langchain_core.prompts import PromptTemplate

initial_planning_prompt = PromptTemplate.from_template("""
You are an expert in CUDA programming, and performance tuning. Your task is to analyze a given CUDA kernel and generate tuning steps. 
Do not implement the optimizations, only provide a detailed plan.
Create steps could be tuned in isolation, and do not require access to the host code.

For the given CUDA kernel:

```CUDA
{kernel_string}
```

Please provide a list of low-level optimization steps.
Do not create steps that would access the host code, only focus on the device code.
""")

breakdown_step_prompt = PromptTemplate.from_template("""
You are an expert in CUDA programming, and performance tuning. 
Your task is to analyze a given CUDA and the optimization step and say if this step needs to be broken down into smaller steps.

For the given CUDA kernel:

```CUDA 
{kernel_string}
```

The optimization step is:
\"\"\"
{current_step}
\"\"\"


Please return True or False if the optimization step needs to be broken down into smaller steps.
Don't make steps too granular, if you can do the optimization in one step, return False and an empty list.
Do not create steps that would access the host code, only focus on the device code.
If the the optimization step needs to be broken down into smaller steps, provide a list of smaller steps.
""")

agent_prompt = PromptTemplate.from_template("""
You are an expert in CUDA programming, and performance tuning. 
Your task is to analyze a given CUDA kernel and apply optimization technique to improve its performance.


Here is the kernel:
```CUDA
{kernel_string}
```
Apply the following optimization technique:
\"\"\"
{optimization_technique}
\"\"\"

Please provide the optimized CUDA kernel code.
Do not change the argument list of the kernel.
You cannot access the host code or any other part of the program.
If the optimized kernel introduces tunable parameters,
explicitly list these parameters.
If no tunable parameters are necessary, do not introduce any.
If the optimization technique is not applicable to the kernel, return the original kernel code.
""")

replan_prompt = PromptTemplate.from_template("""
You are an expert in CUDA programming, GPU optimization, and performance tuning.
Your task is to analyze a given CUDA kernel and optimize it if possible.
If the kernel cannot be optimization further return False and an empty list for steps.
If the kernel can be optimized further, return True and a list of optimization steps.

This is the kernel:
```CUDA
{kernel_string}
```

And here are the optimization steps that you have made so far:
\"\"\"
{past_steps}
\"\"\"

""")

validate_step_prompt = PromptTemplate.from_template("""
Your task is to validate the optimization step for a given CUDA kernel.
Here are is the optimization step:
\"\"\"
{optimization_step}
\"\"\"

Return True if the optimization step is valid, False otherwise.
Here are the criteria for a valid optimization step:
- Optimization must only include device code.
- Optimization must not require access to the host code.
- Optimization must not require any compiler flags.
- Optimization must not require any external tool/libraries.
""")


fix_params_prompt = PromptTemplate.from_template("""
Your task is to look at the given CUDA kernel, existing tunable parameters and new tunable parameters.
Merge the existing and new tunable parameters into a dictionary of tunable parameters.
Choose values for new tunable parameters.
You might also need to adjust the code to use the new tunable parameters.
Remove any defines in the code if a parameter is used


Here are a few examples:
Example 1:
Kernel:
```CUDA
#define BLOCK_SIZE 16 
__global__ void sigmoidActivation(float *in, float *out, int n) {{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {{
        float val = -in[idx];
        out[idx] = 1.0f / (1.0f + __expf(val));
    }}
}}
```
Existing parameters:
```{{'block_size_x': [256]}}```
New tunable parameters:
['BLOCK_SIZE']
Expected output:
{{"kernel_code": "
__global__ void sigmoidActivation(float *in, float *out, int n) {{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {{
        float val = -in[idx];
        out[idx] = 1.0f / (1.0f + __expf(val));
    }}
}}",
"new_params": {{'block_size_x': [32, 64, 128, 256, 512]}}
}}

Example 2:
Kernel:
```CUDA
__global__ void sigmoidActivation(float *in, float *out, int n) {{
    int idx = threadIdx.x + blockDim.x * blockIdx.x * 4;

    #pragma unroll 4
    for (int j = 0; j < 4; ++j) {{
        int index = idx + j * blockDim.x; 
        if (index < n) {{
            out[index] = 1.0f / (1.0f + expf(-in[index]));
        }}
    }}
}}```
Existing parameters:
{{'block_size_x': [32, 64, 128, 256, 512]}}
New tunable parameters:
['UNROLL_FACTOR']
Expected output:
{{
"kernel_code": "
__global__ void sigmoidActivation(float *in, float *out, int n) {{
    int idx = (threadIdx.x + blockDim.x * blockIdx.x) * UNROLL_FACTOR;

    #pragma unroll UNROLL_FACTOR
    for (int i = 0; i < UNROLL_FACTOR; ++i) {{
        if (idx + i < n) {{
            out[idx + i] = 1.0f / (1.0f + expf(-in[idx + i]));
        }}
    }}
}}",
"new_params": {{'block_size_x': [32, 64, 128, 256, 512], 'UNROLL_FACTOR': [2, 4, 8]}}
}}

END OF EXAMPLES

Here is the kernel:
```CUDA
{kernel_string}
```

Here are the existing tunable parameters:
```
{existing_tunable_parameters}
```

Here are the new tunable parameters:
```
{new_tunable_parameters}
```

Notes:
- Make sure that tunable parameters are used in the kernel code and have exactly the same name in code, casing is important.
- Block size is a special case, block size does not have to appear in the kernel code.
- If one of the parameters tunes block size, make sure to name it `block_size_x`, `block_size_y` and `block_size_z`. Other names for block size parameters are not allowed! 
- Tunable parameters must exist as a variable in the kernel code, besides block sizes.
""")

