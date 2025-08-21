Getting started
===============

LLM Kernel Tuner is a framework that helps with tuning and optimizing kernels by utilizing Large Language Models (LLMs).

Here is a simple example of how it can be used for a simple ``matrixMultiply`` kernel:

.. code-block:: python

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

    if __name__ == "__main__":
        kernel_transformer = LLMKernelTransformer(kernel_string, model)
        tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()
        print("Final kernel:")
        print(tuned_kernel.code)
        print("Best params:")
        print(best_params)


| The example above will use OpenAI's gpt-5 model to tune the kernel.
| You can chose any `langchain <https://python.langchain.com/docs/introduction/>`_ chat model, most commonly used models can be found `here <https://python.langchain.com/docs/integrations/chat/>`_.
| By default LLM Kernel Tuner uses :ref:`Autonomous Tuning Strategy <autonomous_tuning_strategy>` and :ref:`Naive Tester Strategy <naive_testing_strategy>`. But you can change these strategies to a :ref:`different tuning strategy <Tuning strategies>`, :ref:`create your own tuning strategy<custom_tuning_strategy>` or :ref:`create your own testing strategy <naive_testing_strategy>`.

   LLM Kernel Tuner uses the `clang` library to parse CUDA kernel code. If the Python `clang` bindings cannot automatically find your `libclang.so` (or equivalent) file, you may need to set the ``LIBCLANG_PATH`` environment variable. For example:

   .. code-block:: bash

      export LIBCLANG_PATH=/usr/lib/llvm-14/lib/libclang.so

   Replace the path with the actual location of the `libclang` shared library on your system.

   Similarly ``clang_args`` can be specified when creating :class:`LLMKernelTransformer <llm_kernel_tuner.LLMKernelTransformer>` like so ``kernel_transformer = LLMKernelTransformer(..., clang_args=['your', 'args'])``. If you are getting the error ``fatal error: '__clang_cuda_runtime_wrapper.h' file not found``, you may need to specify the resource directory: ``clang_args=['-resource-dir', '/usr/lib/clang/18']`` (replace with your actual clang version and path).
