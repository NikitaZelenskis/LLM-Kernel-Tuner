.. _structured_output:

Structured Output
=================

The ``get_structured_llm`` function provides a unified interface for obtaining structured output from different language models, regardless of whether they natively support tool calling, JSON mode, or require a separate prompt-based approach. This is particularly useful for ensuring compatibility with a wide range of LLMs, including those that do not have built-in support for structured data generation.

Why is this needed?
--------------------

Different LLMs have varying capabilities when it comes to generating structured output:

1.  **Tool Calling**: Modern models (e.g., OpenAI's GPT series, Anthropic's Claude) support "tool calling," where the model can be instructed to call a specific function with a given schema. This is the most reliable method for obtaining structured data.

2.  **JSON Mode**: Some models offer a "JSON mode," where they can be constrained to output a JSON object that conforms to a provided schema.

3.  **Default Behavior**: Many models can automatically determine the best approach for structured output based on their capabilities, using their native ``.with_structured_output()`` method without specifying a particular method.

4.  **No Native Support**: Some open-source or older models do not support any of the above methods. They can only generate raw text, which requires additional processing to be converted into a structured format.

The ``get_structured_llm`` function abstracts away these differences, allowing you to work with a consistent interface.

The `get_structured_llm` function and `StructuredOutputType` Enum
------------------------------------------------------------------

.. automodule:: llm_kernel_tuner.structured_output
   :members: get_structured_llm, StructuredOutputType

How it works
------------

The ``get_structured_llm`` function inspects the ``metadata`` attribute of the provided LLM to determine the appropriate method for generating structured output.

- If ``structured_output_type`` is set to ``TOOL_CALLING``, it uses the native ``.with_structured_output()`` method with function calling.
- If ``structured_output_type`` is set to ``JSON_SCHEMA``, it uses the native ``.with_structured_output()`` method with JSON mode.
- If ``structured_output_type`` is set to ``DEFAULT`` (the default), it uses the native ``.with_structured_output()`` method without specifying a particular method, allowing the model to choose the optimal strategy.
- If ``structured_output_type`` is set to ``SEPARATE_REQUEST``, it falls back to the ``StructuredOutputEmulator``, which uses a separate prompt to format the raw text output into the desired Pydantic object.
- If ``structured_output_type`` is set to ``HYBRID_JSONIFY``, it uses a two-step process where the first LLM call generates raw text, and a second call uses the model's native structured output capability to format the text into JSON. This is useful for LLMs that don't allow thinking when structured output is enabled.

The ``structured_output_type`` is typically configured through the :class:`LLMKernelTransformer <llm_kernel_tuner.LLMKernelTransformer>` constructor parameter, which automatically sets the appropriate metadata on the LLM instance.

Example Usage
-------------

.. code-block:: python

    from llm_kernel_tuner import LLMKernelTransformer
    from llm_kernel_tuner.structured_output import StructuredOutputType
    from langchain_openai import ChatOpenAI

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

    # Configure LLM to use tool calling for structured output
    llm = ChatOpenAI(model="gpt-4o")
    kernel_transformer = LLMKernelTransformer(
        kernel_string, 
        llm,
        structured_output_type=StructuredOutputType.TOOL_CALLING
    )

    # Configure LLM to use default behavior
    kernel_transformer_default = LLMKernelTransformer(
        kernel_string, 
        llm,
        structured_output_type=StructuredOutputType.DEFAULT
    )

    # Configure LLM to use separate request emulation
    kernel_transformer_emulated = LLMKernelTransformer(
        kernel_string, 
        llm,
        structured_output_type=StructuredOutputType.SEPARATE_REQUEST
    )

    # Configure LLM to use hybrid jsonify
    kernel_transformer_hybrid = LLMKernelTransformer(
        kernel_string, 
        llm,
        structured_output_type=StructuredOutputType.HYBRID_JSONIFY
    )
