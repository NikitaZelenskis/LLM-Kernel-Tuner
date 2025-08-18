# os.environ['LIBCLANG_PATH'] = '/data/s2622157/llvm/LLVM-20.1.3-Linux-X64/lib/libclang.so' # if you want to use a different libclang.so file than the fefault one
import logging

# make sure llama-cpp-python is installed before running this script `pip install llama-cpp-python`` 
from langchain_community.chat_models import ChatLlamaCpp
from llm_kernel_tuner import LLMKernelTransformer

if __name__ == "__main__":
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

    logging.basicConfig(level=logging.DEBUG)

    gguf_path = "path/to/your.gguf"

    llm = ChatLlamaCpp(
        model_path=gguf_path,
        n_gpu_layers=-1,
        n_ctx=10000,
        max_tokens=5000,
        # verbose=True,
    )

    # you can change resource dir to a non default one 
    # kernel_transformer = LLMKernelTransformer(kernel_string, llm, clang_args=['-resource-dir', '/data/s2622157/llvm/LLVM-20.1.3-Linux-X64/lib/clang/20'])
    kernel_transformer = LLMKernelTransformer(kernel_string, llm)
    tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()
    print("Final kernel:")
    print(tuned_kernel.code)
    print("Best params:")
    print(best_params)
