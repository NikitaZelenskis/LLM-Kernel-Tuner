Using different LLM models
==========================

LLM Kernel Tuner is designed to be highly flexible, allowing you to use any LLM model that is compatible with LangChain. This includes both local and cloud-based models, as well as models from providers such as OpenAI, Hugging Face, Anthropic, and more.

Since LLM Kernel Tuner leverages LangChain's standardized interface, switching between models requires minimal changes to your configuration. Whether you're working with an API-based model or running an open-source model locally, you can easily integrate it into your workflow.

In the following sections, we will cover how to configure different types of models, including:

* API-based models (e.g., OpenAI, Anthropic)
* Self-hosted models running through llama.cpp

By following these guidelines, you can optimize your LLM setup to match your specific needs, whether it's performance, cost, or privacy considerations.

Handling Structured Output
^^^^^^^^^^^^^^^^^^^^^^^^^^

While LLM Kernel Tuner works best with models that natively support structured output (also known as tool calling or function calling), it includes a robust fallback system for models that don't. This is managed by the ``get_structured_llm`` utility function.

The library can handle structured output in four ways, configured via the ``structured_output_type`` parameter in the :class:`LLMKernelTransformer <llm_kernel_tuner.LLMKernelTransformer>` constructor:

1.  ``TOOL_CALLING``: Uses the model's native tool-calling ability.
2.  ``JSON_SCHEMA``: Uses the model's native ability to output JSON that conforms to a schema.
3.  ``DEFAULT``: Uses the model's default structured output behavior, allowing the model to automatically determine the best approach.
4.  ``SEPARATE_REQUEST``: For models with no native support, it uses an emulation layer. This sends a second, specially crafted prompt to the LLM, asking it to format its previous raw output into the required JSON structure.

This allows you to use a wider variety of models, including many open-source ones, without changing your core logic.

**Example: Configuring Structured Output**

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

    # This model supports tool calling natively, but force it to use emulation
    model = ChatOpenAI(model="gpt-4o")
    
    kernel_transformer = LLMKernelTransformer(
        kernel_string,
        model,
        structured_output_type=StructuredOutputType.SEPARATE_REQUEST  # Force emulation
    )


For more detailed information on this mechanism, see the :ref:`structured_output` documentation.

OpenAI
^^^^^^

LLM Kernel Tuner uses OpenAI's GPT-4o model by default and therefore ``langchain-openai`` package should already be installed.
However if you want to change the model in use you can do so like this:

.. code-block:: python

    from llm_kernel_tuner import LLMKernelTransformer
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

    # All models can be found here: https://platform.openai.com/docs/models
    openAI_model_name = "gpt-4.5-preview" 

    model = ChatOpenAI(model_name=openAI_model_name)

    kernel_transformer = LLMKernelTransformer(kernel_string, model)
    tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()

| The example above will use the latest GPT-4.5 preview model to tune ``matrixMultiply`` kernel defined in ``kernel_string``.
| Make sure that ``OPENAI_API_KEY`` is set.

Anthropic
^^^^^^^^^

| To use the Anthropic models you will first need to install ``langchain-antrhopic`` package.
| Also make sure that ``ANTHROPIC_API_KEY`` environment variable is set.

.. code-block:: python

    from llm_kernel_tuner import LLMKernelTransformer
    from langchain_anthropic import ChatAnthropic

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

    # All models can be found here: https://docs.anthropic.com/en/docs/about-claude/models/all-models
    anthropic_model_name = "claude-3-7-sonnet-latest" 

    model = ChatAnthropic(model_name=anthropic_model_name, timeout=None, stop=None)

    kernel_transformer = LLMKernelTransformer(kernel_string, model)
    tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()

The example above will use Anthropic's latest version of Claude Sonnet 3.7 to tune ``matrixMultiply`` kernel defined in ``kernel_string``.


.. Hugging Face
.. ^^^^^^^^^^^^
.. langchain doesn't seem to work with hf :/ https://github.com/langchain-ai/langchain/discussions/26321



llama.cpp with Python Bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    from llm_kernel_tuner import LLMKernelTransformer
    from langchain_community.chat_models import ChatLlamaCpp


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

    gguf_path = "/path/to/your/model.guff"

    model = ChatLlamaCpp(
        model_path=gguf_path,
        n_gpu_layers=-1,
        n_ctx=10000,
        max_tokens=5000,
    )

    kernel_transformer = LLMKernelTransformer(kernel_string, model)
    tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()

    print("Final kernel:")
    print(tuned_kernel.code)
    print("Best params:")
    print(best_params)


vLLM
^^^^

Here is a full example of how you can setup LLM kernel tuner to work with vLLM 
by starting vLLM openAI server in a separate process and waiting untill server is ready.
This example also handles errors in the subprocess.

.. code-block:: python

    import subprocess
    import requests
    import time
    import atexit
    import signal
    import sys
    import os
    from typing import Optional

    # --- Configuration Constants ---
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    PORT = "8000"
    MAX_TOKENS = 10000

    VLLM_SERVER_URL = f"http://localhost:{PORT}/v1"
    SERVER_READINESS_TIMEOUT = 300 # seconds (5 minutes)
    CLEANUP_WAIT_TIMEOUT = 5 # seconds

    # --- Global variable to hold the subprocess ---
    vllm_proc: Optional[subprocess.Popen] = None

    # --- Cleanup Function ---
    def cleanup_vllm_subprocess():
        """Terminates the vLLM subprocess if it's running."""
        global vllm_proc
        if vllm_proc and vllm_proc.poll() is None: # Check if process exists and is running
            print("\nAttempting to terminate vLLM subprocess...", flush=True)
            try:
                # Send SIGTERM first (graceful shutdown)
                vllm_proc.terminate()
                vllm_proc.wait(timeout=CLEANUP_WAIT_TIMEOUT)
                print("vLLM subprocess terminated gracefully.", flush=True)
            except subprocess.TimeoutExpired:
                print(f"vLLM subprocess did not terminate gracefully after {CLEANUP_WAIT_TIMEOUT}s, forcing kill...", flush=True)
                vllm_proc.kill() # Send SIGKILL (force kill)
                vllm_proc.wait() # Wait for kill confirmation
                print("vLLM subprocess killed.", flush=True)
            except Exception as e:
                print(f"Error during vLLM cleanup: {e}", flush=True)
            finally:
                vllm_proc = None # Ensure we don't try cleanup again
        elif vllm_proc:
            # Process already finished, just clear the variable
            print("vLLM subprocess already terminated.", flush=True)
            vllm_proc = None


    # --- Signal Handling ---
    def handle_signal(signum, frame):
        """Handles termination signals like SIGINT (Ctrl+C) and SIGTERM."""
        signal_name = signal.Signals(signum).name
        print(f"\nReceived signal {signal_name}. Cleaning up...", flush=True)
        cleanup_vllm_subprocess()
        # Exit the script after cleanup. Use a non-zero exit code for signals.
        sys.exit(128 + signum) # Standard convention for exit code after signal

    # --- Register Cleanup Hooks ---
    atexit.register(cleanup_vllm_subprocess)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # --- Helper Function for Server Readiness ---
    def wait_for_server_ready(proc: subprocess.Popen, url: str, timeout: int) -> bool:
        """Polls the server URL until it's ready or timeout occurs."""
        print(f"Waiting up to {timeout} seconds for vLLM server at {url}...", flush=True)
        start_time = time.time()
        check_url = f"{url}/models"

        while time.time() - start_time < timeout:
            # Check if the subprocess terminated unexpectedly
            if proc.poll() is not None:
                print(f"❌ vLLM process exited prematurely with code: {proc.returncode}", flush=True)
                # You might want to capture and print stderr from the Popen call
                # if you need more info on why it failed
                return False # Server failed to start

            try:
                response = requests.get(check_url, timeout=2)
                if response.status_code == 200 and '"id"' in response.text:
                    print("✅ vLLM server is ready!", flush=True)
                    return True
                else:
                    # Log unexpected status codes if needed
                    # print(f"Server status check: {response.status_code}, waiting...", flush=True)
                    pass # Often just waiting for 200
            except requests.exceptions.ConnectionError:
                # Server not up yet, expected during startup
                # print("Server connection refused, waiting...", flush=True) # Can be verbose
                pass
            except requests.exceptions.Timeout:
                print("Server status check timed out, retrying...", flush=True)
            except Exception as e:
                print(f"Error checking server status: {e}", flush=True)
                # Depending on the error, you might want to break or continue

            time.sleep(3) # Polling interval

        print(f"❌ vLLM server did not become ready within {timeout} seconds.", flush=True)
        return False

    # --- Main Execution Logic ---
    def main():
        """Starts the vLLM server, waits for it, runs the LLM interaction, and ensures cleanup."""
        global vllm_proc # Declare we intend to modify the global variable

        try:
            print("Starting vLLM server subprocess...", flush=True)
            cmd = [
                sys.executable, # Use the same python interpreter
                "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL_NAME,
                "--port", PORT,
            ]
            print(f"Executing command: {' '.join(cmd)}", flush=True)

            # Start the subprocess
            # Consider capturing stderr if you want to debug vLLM startup issues:
            # stderr=subprocess.PIPE, text=True
            vllm_proc = subprocess.Popen(cmd)
            print(f"vLLM subprocess started with PID: {vllm_proc.pid}", flush=True)

            # Wait for the server to become ready
            if not wait_for_server_ready(vllm_proc, VLLM_SERVER_URL, SERVER_READINESS_TIMEOUT):
                print("❌ Exiting due to server startup failure.", flush=True)
                # Cleanup will be triggered by atexit/signal handlers upon exit
                sys.exit(1) # Exit indicating an error

            # --- If server is ready, proceed with Langchain/LLM tasks ---
            print("\n--- Initializing Langchain and LLM Tuner ---", flush=True)
            # Lazy import after server start to avoid importing heavy libraries if server fails
            from langchain_openai import ChatOpenAI
            from llm_kernel_tuner import LLMKernelTransformer

            model = ChatOpenAI(
                model=MODEL_NAME,
                api_key="EMPTY",
                base_url=VLLM_SERVER_URL,
                max_tokens=MAX_CONTEXT_LENGTH,
            )

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

            print("\n--- Starting Kernel Tuning ---", flush=True)
            tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()

            print("\n--- Tuning Complete ---", flush=True)
            print("\nFinal tuned kernel code:")
            print(tuned_kernel.code)
            print("\nBest parameters found:")
            print(best_params)

            print("\nScript finished successfully.", flush=True)

        except FileNotFoundError as e:
            print(f"❌ Error starting subprocess: {e}. Is '{sys.executable}' correct or is 'vllm' installed in this environment?", file=sys.stderr, flush=True)
            # Cleanup already handled by atexit/signal
            sys.exit(1)
        except ImportError as e:
            print(f"❌ Error importing required library: {e}. Please ensure langchain_openai and llm_kernel_tuner are installed.", file=sys.stderr, flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ An unexpected error occurred in the main script: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc() # Print full traceback for unexpected errors
            # Cleanup already handled by atexit/signal
            sys.exit(1) # Exit indicating an error

    # --- Script Entry Point ---
    if __name__ == "__main__":
        main()

If you have the server already running you could omit the start subprocess and just connect to the server directly

.. code-block:: python

    from langchain_openai import ChatOpenAI
    from llm_kernel_tuner import LLMKernelTransformer

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"


    inference_server_url = "http://localhost:8000/v1"


    model = ChatOpenAI(
        model=model_name,
        api_key="EMPTY",
        base_url=inference_server_url,
        max_tokens=50000,
    )



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

You can find the list of all supported vLLM models here: `<https://docs.vllm.ai/en/latest/models/supported_models.html#model-support-policy>`_
