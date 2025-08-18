from langchain_core.prompts import PromptTemplate

system_prompt = """
You are an expert in CUDA programming and GPU optimization.
User will provide you with a CUDA kernel and your task is to optimize it for performance while maintaining correctness.
Only output the device code. 
Do not modify or access the host code.
Do not change function arguments.
If there are any tunable parameters in the code you provided return them aswell.
"""

user_prompt = PromptTemplate.from_template("""
Please optimize the following CUDA kernel:
```CUDA
{kernel_string}
```""")

fix_params_prompt = PromptTemplate.from_template("""
Your task is to give values for tunable parameters and add them to the code if needed.
First look at the given CUDA kernel and a list of tunable parameters.
Create a dictionary of <string, List<int>> where key is the name of a tunable parameter and value is a list of values for the tunable parameter
Choose apropriate values for tunable parameters that will fit the tunable parameter.
You might also need to adjust the code to use the new tunable parameters.

Here is the kernel:
```CUDA
{kernel_string}
```

Here are the existing tunable parameters:
```
{tunable_parameters}
```


Notes:
- Make sure that tunable parameters are used in the kernel code, casing is important.
- Tunable parameters should be in lower case.
- Tunable parameters should be used in the kernel code.
- If one of the parameters tunes block size, make sure to name it block_size_x, block_size_y or block_size_z.
""")