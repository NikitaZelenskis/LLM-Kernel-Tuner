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
MAX_CONTEXT_LENGTH = "15000"
MAX_TOKENS = 10000

DOWNLOAD_DIR = os.path.expanduser("/data/s2622157/vllm")
TENSOR_PARALLEL_SIZE = "4"
VLLM_SERVER_URL = f"http://localhost:{PORT}/v1"
SERVER_READINESS_TIMEOUT = 300 # seconds (5 minutes) it can take a some time for large models to load into gpus
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
            # if you need more info on why it failed (e.g., invalid DOWNLOAD_DIR)
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

    # Removed the directory check/creation block.
    # Ensure DOWNLOAD_DIR exists and is writable before running.

    try:
        print("Starting vLLM server subprocess...", flush=True)
        cmd = [
            sys.executable, # Use the same python interpreter
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--port", PORT,
            "--download-dir", DOWNLOAD_DIR,
            "--tensor-parallel-size", TENSOR_PARALLEL_SIZE,
            "--max-model-len", MAX_CONTEXT_LENGTH,
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
            max_tokens=MAX_TOKENS, 
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