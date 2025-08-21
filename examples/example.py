import os
from llm_kernel_tuner import LLMKernelTransformer
from langchain_openai import ChatOpenAI

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# Check if the API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Please set the OPENAI_API_KEY environment variable to your OpenAI API key.")
    exit()

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

if __name__ == "__main__":
    kernel_transformer = LLMKernelTransformer(kernel_string, model, clang_args=['-resource-dir', '/usr/lib/clang/18'])
    tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()
    print("Final kernel:")
    print(tuned_kernel.code)
    print("Best params:")
    print(best_params)
