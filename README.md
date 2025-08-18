# LLM Kernel Tuner

A package for automated kernel tuning with LLMs.

LLM Kernel Tuner is a framework that helps with tuning and optimizing kernels by utilizing Large Language Models (LLMs).

## Features

*   **Automated Kernel Tuning**: Automatically tune and optimize your kernels using LLMs.
*   **Extensible**: Easily extend the framework with your own tuning and testing strategies.
*   **Flexible**: Supports various LLMs through `langchain`.

## Installation

You can install LLM Kernel Tuner from the local directory after you have cloned it:

```bash
pip install .
```

Alternatively, you can install it directly from the GitHub repository:

```bash
pip install git+https://github.com/NikitaZelenskis/LLM-Kernel-Tuner.git
```

## Getting Started

Here is a simple example of how to use LLM Kernel Tuner for a simple `matrixMultiply` kernel:

```python
from llm_kernel_tuner import LLMKernelTransformer
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model_name='gpt-5')

kernel_string = """
__global__ void matrixMultiply(float *A, float *B, float *C, int A_width, int A_height, int B_width) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col < B_width && row < A_height) {
        float sum = 0;
        for (int k = 0; k < A_width; ++k) {
            sum += A[row * A_width + k] * B[k * B_width + col];
        }
        C[row * B_width + col] = sum;
    }
}
"""

kernel_transformer = LLMKernelTransformer(kernel_string, model)
tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()
print("Final kernel:")
print(tuned_kernel.code)
print("Best params:")
print(best_params)
```

## Documentation

For more detailed information, please refer to the [documentation](https://nikitazelenskis.github.io/LLM-Kernel-Tuner/).

## License

This project is licensed under the MIT License. Please note that the license is not explicitly specified in the `pyproject.toml` file.

## Examples

You can find more examples in the `examples` directory:
*   [Llama CPP Example](examples/llama_cpp_example.py)
*   [vLLM Example](examples/vllm_example.py)
