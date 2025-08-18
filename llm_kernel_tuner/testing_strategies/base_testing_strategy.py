from llm_kernel_tuner.tunable_kernel import TunableKernel
from llm_kernel_tuner.retry import RetryPolicy, InvalidTest, default_tester_retry_policy, SharedMemorySizeExceededError
from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from llm_kernel_tuner.kernel_test import KernelTest, TestInputType
from langchain_core.messages import BaseMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from kernel_tuner import run_kernel
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, TypedDict, Dict, Any, overload
from multiprocessing import shared_memory, resource_tracker
import numpy as np
import os
import subprocess
import tempfile
import uuid
import pickle
import sys
import selectors
import time

logger = get_logger(__name__)


class SubprocessTimeoutError(Exception):
    pass


_RETRY_POLICY_DEFAULT_SENTINEL = object()  # Sentinel object for default retry_policy


class BaseTestingStrategy(ABC):
    """Base class for generating testing strategies for kernel.
    
    This abstract class defines the interface for testing strategies that generate tests for a given kernel.
    Implementations of this class should generate tests and store them in ``state["tests"]``.

    Args:
        retry_policy (RetryPolicy, optional): The retry policy to use for the testing strategy. Defaults to `default_tester_retry_policy`
        max_data_size (int, optional): The maximum size of the data that can be passed to the kernel. Defaults to 2GB.

    Note:
        | Subclasses must implement the :func:`create_graph <BaseTestingStrategy.create_graph>` method.
        | :class:`llm_kernel_tuner.tuning_state.State` will be passed to this graph.
    
    See Also:
        :ref:`custom_testing_strategy` for example usage.
    """
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, retry_policy: Optional[RetryPolicy]) -> None: ...

    @overload
    def __init__(self, retry_policy: Optional[RetryPolicy], max_data_size: int) -> None: ...


    def __init__(self, retry_policy: Any = _RETRY_POLICY_DEFAULT_SENTINEL, max_data_size: int = 5 * 1024**3): #5GB
        self.llm: Optional[BaseChatModel] = None
        self.template_file_name = "LLMTestExtraction.template.py"
        module_dir = os.path.dirname(__file__)
        self.template_file_path = os.path.join(module_dir, '.', self.template_file_name)
        self.max_data_size = max_data_size
        
        if retry_policy is _RETRY_POLICY_DEFAULT_SENTINEL:
            self.retry_policy: Optional[RetryPolicy] = default_tester_retry_policy
        else:
            # Based on the overloads, if retry_policy is not the sentinel,
            # it must conform to Optional[RetryPolicy].
            self.retry_policy: Optional[RetryPolicy] = retry_policy

    @abstractmethod
    def create_graph(self, llm: BaseChatModel) -> CompiledStateGraph:
        """This method must be implemented.
        

        Args:
            llm (BaseChatModel): LLM model that user wants to use for tuning. 

        Returns:
            CompiledStateGraph: langchain graph that will be called for tuning. ``llm_kernel_tuner.tuning_state.State`` will be passed to this graph. 
        """
        pass


    def get_test_from_code(self, kernel: TunableKernel, code: str, params: Dict[str, Any], timeout: float) -> KernelTest:
        """Generates a kernel test case from provided Python code.

        Executes the given Python ``code`` string, wrapped in a template, in a
        separate process to obtain input data. The provided `code` should define
        a variable named ``input_data`` containing a list of numpy arrays for the kernel
        arguments. The wrapping template handles saving this data to a temporary
        file and printing its path.

        Then, this method runs the ``kernel`` using the obtained inputs and the
        specified tuning ``params``. Finally, it packages the inputs, corresponding
        kernel outputs, and the derived problem size into a KernelTest object.

        Args:
            kernel (TunableKernel): The TunableKernel object representing the kernel to test.
            code (str): A string containing Python code that defines a variable named
                ``input_data`` as a list of numpy arrays representing the kernel
                input arguments. Here is an example of how ``code`` is supposed to look like:
                
                .. code-block:: python

                    import numpy as np
                    size = 10000000

                    a = np.random.randn(size).astype(np.float32)
                    b = np.random.randn(size).astype(np.float32)
                    c = np.zeros_like(a)
                    n = np.int32(size)

                    input_data = [c, a, b, n]
                
            params (Dict[str, Any]): A dictionary containing the tuning parameters to be used
                    when running the kernel to generate outputs.
            timeout (float, optional): Timeout in seconds for the process to generate input. Default is 120.

        Returns:
            A KernelTest object containing the generated input arguments, the
            corresponding kernel output arguments, and the problem size used
            for the kernel execution.

        Raises:
            InvalidTest: If the provided ``code`` fails to execute successfully
                         within the template, if the template execution does not
                         produce the expected output file path, or if the kernel
        """
        full_code = self._wrap_code(code)
        result = self._run_generated_code(full_code, timeout)
        if result is not None:
            return self._get_test_from_input(kernel, result, params)
        raise InvalidTest("Could not generate input from the provided code")


    def _wrap_code(self, code: str) -> str:
        """
        Wraps the provided test code within a template for execution.
    
        Takes generated test code and wraps it within a predefined template file 
        that contains the necessary setup for test execution. 

        Args:
            code (str): A string containing the test code to be wrapped.
            
        Returns:
            A string containing the complete code ready for execution, with the 
            input code properly indented and inserted into the template.
        """
        with open(self.template_file_path) as file:
            template = file.read()

        # remove print statements from the code
        lines = [line for line in code.strip().split('\n') if not line.strip().startswith('print(')]
        cleaned_code = '\n'.join(lines)

        indented_code = '\n    '.join(cleaned_code.strip().split('\n'))
        full_code = template.format(generated_code=indented_code)
        return full_code
    

    def _run_generated_code(self, code: str, timeout: float = 120) -> Optional[List[TestInputType]]:
        """Creates a temporary Python file with the provided code, executes it in a
        separate process with a timeout, and collects the input arrays using shared memory.

        Args:
            code (str): A string containing the complete Python code to be executed.
            timeout (float, optional): Timeout in seconds for the process to generate input. Default is 120.

        Returns:
            Optional[List[np.ndarray]]: A list containing the input as numpy arrays
            if execution is successful, or None if the execution fails or times out.

        Note:
            - Uses a handshake protocol with the child process to avoid deadlocks.
            - The parent process creates and owns the shared memory block.
        """
        tmp_file_name = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as tmp_file:
            tmp_file_name = tmp_file.name
            tmp_file.write(code)

        process = None
        shm = None
        input_data = None
        start_time = time.monotonic()

        try:
            process = subprocess.Popen(
                [sys.executable, tmp_file_name],
                text=True,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
            )

            if process.stdout is None or process.stdin is None or process.stderr is None:
                logger.error("Failed to create subprocess pipes")
                return None
            
            try:
                # Read data size from child
                remaining_time = timeout - (time.monotonic() - start_time)
                if remaining_time <= 0:
                    raise TimeoutError("Timeout occurred before receiving data size from child.")
                
                data_size_str = self._read_line_with_timeout(process.stdout, remaining_time)
                data_size = int(data_size_str)

                if data_size > self.max_data_size:
                    raise SharedMemorySizeExceededError(data_size, self.max_data_size)

                shm = self._create_shared_memory(data_size, process)

                # Wait for child to signal completion
                remaining_time = timeout - (time.monotonic() - start_time)
                if remaining_time <= 0:
                    raise TimeoutError("Timeout occurred before child signaled completion.")

                input_data = self._process_child_signal(shm, process, data_size, remaining_time)
                
            except (ValueError, IndexError) as e:
                error = process.stderr.read()
                logger.error(f"Could not parse data from child process. Error: {e}. Stderr: {error}")
                return None
            except TimeoutError as e:
                # This is the specific exception for our timeout logic
                logger.error(f"Subprocess timed out after {time.monotonic() - start_time:.2f} seconds. Killing process.")
                logger.error(f"Timeout details: {e}")
                # The exception will be re-raised after the finally block.
                raise
            except BrokenPipeError:
                # This can happen if the child process exits unexpectedly before the handshake is complete
                error = process.stderr.read()
                logger.error(f"Communication pipe with child process broke. Child may have crashed. Stderr: {error}")
                return None

        finally:
            self._cleanup_resources(process, shm, tmp_file_name)

        return self._process_data(input_data)


    def _process_data(self, input_data: Optional[List[Any]]) -> Optional[List[TestInputType]]:
        if input_data:
            for i, el in enumerate(input_data):
                if isinstance(el, int):
                    input_data[i] = np.int32(el)
                elif isinstance(el, float):
                    input_data[i] = np.float32(el)
        
        return input_data


    def _cleanup_resources(self, child_process: Optional[subprocess.Popen[str]], shm: Optional[shared_memory.SharedMemory], tmp_file_name: str):
        if child_process:
            if child_process.poll() is None: # If process is still running
                try:
                    child_process.stdin.close() # type: ignore as it is checked in caller function
                except (IOError, BrokenPipeError):
                    pass # Child may have already exited
                
                try:
                    child_process.wait(timeout=5)#wait for the process to finish
                except:
                    # Try graceful termination first
                    logger.warning(f"Process {child_process.pid} did not terminate naturally, sending SIGTERM.")
                    child_process.terminate()
                    try:
                        child_process.wait(timeout=5)  # Give it time to clean up
                    except subprocess.TimeoutExpired:
                        # Force kill as last resort
                        logger.warning(f"Process {child_process.pid} did not respond to SIGTERM, sending SIGKILL.")
                        child_process.kill()
                        try:
                            child_process.wait(timeout=10)  # Give SIGKILL time to work
                        except subprocess.TimeoutExpired:
                            logger.error(f"Process {child_process.pid} survived SIGKILL - likely stuck in kernel.")
                            # At this point, we can't do anything more


        if shm:
            shm_name = shm.name
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            finally:
                resource_tracker.unregister(f"/{shm_name}", 'shared_memory')

        
        if tmp_file_name and os.path.exists(tmp_file_name):
            os.remove(tmp_file_name)

    def _create_shared_memory(self, data_size: int, child_process: subprocess.Popen[str]) -> shared_memory.SharedMemory:
        shm_name = f"llm-kernel-tuner-shm-{uuid.uuid4()}"
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=data_size)
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise
        # Send shared memory name to child
        child_process.stdin.write(f"{shm_name}\n")# type: ignore as it is checked in caller function
        child_process.stdin.flush()# type: ignore as it is checked in caller function

        return shm

    def _process_child_signal(self, shm: shared_memory.SharedMemory, child_process: subprocess.Popen[str], data_size: int, remaining_time: float) -> Optional[List[Any]]:
        completion_signal = self._read_line_with_timeout(child_process.stdout, remaining_time)
        
        # wait until child process is done writing to the shared memory
        # when its done it send "DONE" signal
        if completion_signal != "DONE":
            error_output = child_process.stderr.read()# type: ignore as it is checked in caller function
            logger.error(f"Child process failed. Signal: '{completion_signal}'. Stderr: {error_output}")
            return None
        
        input_data: List[Any] = pickle.loads(shm.buf[:data_size])

        return input_data

    def _read_line_with_timeout(self, stream, timeout: float) -> str:
        """Reads a line from a stream with a timeout."""
        # selectors is the modern, high-level API for non-blocking I/O
        with selectors.DefaultSelector() as selector:
            selector.register(stream, selectors.EVENT_READ)
            # Wait for the stream to be ready for reading, with a timeout
            if selector.select(timeout=timeout):
                return stream.readline().strip()
            else:
                # The select call timed out
                raise SubprocessTimeoutError(f"Read operation timed out after {timeout:.2f} seconds.")


    def _get_test_from_input(self, kernel: TunableKernel, input_data: List[TestInputType], params: Dict[str, Any]) -> KernelTest:     
        """Runs the kernel and extracts the output variables based on the provided list.

        Args:
            kernel: The kernel to run.
            input_data: The input arguments for the kernel.
            params: Tuning parameters for the kernel.

        Returns:
            A list of the same length as the kernel arguments. Contains the output
            numpy arrays at the positions corresponding to the output variables,
            and None otherwise. Returns an empty list if kernel execution fails.
        """
        # run_kernel returns a list of all arguments after kernel execution
        params_flat:Dict[str, Union[int,float]] = {k: v[0] for k, v in params.items()}
        
        problem_size = self._problem_size_to_tuple(kernel, input_data)

        try:
            results: List[TestInputType] = run_kernel(kernel.kernel_info.name, kernel.code, problem_size, input_data, params_flat)
        except Exception as e:
            logger.error(f"Kernel execution failed during output generation: {e}")
            raise InvalidTest(e)

        # Initialize the output list with Nones, matching the number of kernel arguments
        kernel_outputs: List[Optional[TestInputType]] = [None] * len(kernel.kernel_info.args)


        # Populate the output list with actual results for specified output variables
        for var_name in kernel.kernel_info.output_variables:
            pos = kernel.get_arg_position(var_name)
            if pos is not None:
                kernel_outputs[pos] = results[pos]
            else:
                logger.warning(f"Output variable '{var_name}' specified by LLM not found in kernel arguments.")

        return KernelTest(input_data, kernel_outputs, problem_size)

        

    def _problem_size_to_tuple(self, kernel: TunableKernel, input: List[TestInputType]) -> Union[int, Tuple[int, ...]]:
        """
        Determines the problem size for kernel execution based on kernel metadata and input arguments.

        Args:
            kernel: The TunableKernel object containing metadata about the kernel.
            input: A list of numpy arrays representing the input arguments to the kernel.

        Returns:
            Either an integer (if problem size is defined by a single variable) or a tuple of integers
            (if problem size is defined by multiple variables).

        Raises:
            ValueError: If a problem size variable specified in the kernel metadata is not found
                        in the kernel arguments or if the corresponding input argument is not a scalar integer.
        """
        problem_size_vars = kernel.kernel_info.problem_size
        problem_size_values = []

        for var_name in problem_size_vars:
            pos = kernel.get_arg_position(var_name)
            if pos is None:
                raise ValueError(f"Problem size variable '{var_name}' not found in kernel arguments.")

            # Ensure the input argument exists and is a scalar or size-1 array
            if pos >= len(input):
                raise ValueError(f"Input argument for problem size variable '{var_name}' at position {pos} is missing.")

            input_arg = input[pos]

            # Extract the scalar integer value
            if isinstance(input_arg, (np.number, int, float)):
                value = int(input_arg)
            elif isinstance(input_arg, np.ndarray) and input_arg.size == 1:
                value = int(input_arg.item())
            else:
                raise ValueError(f"Input argument for problem size variable '{var_name}' must be a scalar integer or a size-1 numpy array. Got type {type(input_arg)} with shape {getattr(input_arg, 'shape', 'N/A')}.")

            problem_size_values.append(value)

        if len(problem_size_values) == 1:
            return problem_size_values[0]
        elif len(problem_size_values) > 1:
            return tuple(problem_size_values)
        else:
            # This case should ideally not happen if kernel metadata is valid
            raise ValueError("Problem size definition is empty in kernel metadata.")