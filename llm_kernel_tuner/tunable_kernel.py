from llm_kernel_tuner.kernel_test import KernelTest, TestInputType
from llm_kernel_tuner.retry import CompileErrorError, FailedTestsError
from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Callable, Union
from kernel_tuner import tune_kernel, run_kernel
import numpy as np
from queue import Empty

# embedded in try block to be able to generate documentation
# and run tests without pycuda installed
try:
    import pycuda.autoinit
    import pycuda.tools  # Add this import
    pycuda_available = True
except ImportError:
    pycuda_available = False


logger = get_logger(__name__)


 
class TunableKernelInfo:
    """Stores shared information about a tunable kernel that does not change during the transformation process.

    This class holds metadata that is common across different instances or
    copies of a `TunableKernel`, avoiding redundant storage.
    """
    def __init__(self
                 ):
        """Arguments of the function.

        Examples: 
            `__global__ void vectorAdd(float *A, float *B, float *C, int n)` ->  `{"A": "float *", "B": "float *", "C": "float *", "n": "int"}`"""
        self.args: Dict[str, str]

        """Name of the function in code.
        
        Examples:
            `__global__ void vectorAdd(float *A, float *B, float *C, int n)` ->  `vectorAdd`"""
        self.name: Optional[str]

        """Description of the kernel.""" 
        self.description: str

        """Describles which variables and in what way are the used as problem size for the tuning process.
        Mostly used by tests. 
        """
        self.problem_size: List[str]
        """Describes which variables are being used as output of the kernel.
        This is used for extracting the output of the kernel after it has been run.

        Example:

        .. code-block:: CUDA

            __global__ void vector_add(float *c, float *a, float *b, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i<n) {
                    c[i] = a[i] + b[i];
                }
            }

        The following kernel stores the output variable c, so ``output_variables`` will be ``[c]`` 
        """
        self.output_variables: List[str]
        """Time per test in seconds"""
        self.time_per_test = 15
        """Minimum performance improvement threshold as a percentage (default 0.5%)"""
        self.performance_threshold: float = 0.5
        self.device: int
        self.cuda_gpu_arch: str
        self.clang_args: List[str]




class TuneResult:
    def __init__(self,
                 time: Optional[int] = None,
                 best_tune_params: Dict[str, Any] = {}):
        self.time = time
        self.best_tune_params = best_tune_params

class TunableKernel:
    """
    Kernel that can be used for tuning.

    Args:
        code (str): The kernel code that will be tuned.
        kernel_info (TunableKernelInfo): Object that stores general information about the kernel.
        This object is shared between internal copies of TunableKernel during tuning session.
    """

    def __init__(self, code: str, kernel_info: TunableKernelInfo):
        self.code: str = code
        """Info that is shared between copies of tunable kernel"""
        self.kernel_info = kernel_info

        """Best performing tune parameters"""
        self.best_tune_params:Optional[Dict[str, Any]] = None
        """Time elapesd for the best performing tune parameters"""
        self.best_time: Optional[int] = None


    def get_arg_position(self, arg:str) -> Optional[int]:
        """
        Returns the position of the argument. 
        
        Args:
            arg (str): name of the argument.
        
        Returns:
            int | None: Returns the zero indexed position of the argument or ``None`` if the argument does not exist.
        """
        return list(self.kernel_info.args.keys()).index(arg)
        
    def copy(self) -> 'TunableKernel':
        """
        Creates a full copy of the TunableKernel instance.
        
        Returns:
            TunableKernel: A new instance with the same attribute values.
        """
        kernel_copy = TunableKernel(
            self.code,
            self.kernel_info
        )
        kernel_copy.best_tune_params = self.best_tune_params.copy() if self.best_tune_params else None
        kernel_copy.best_time = self.best_time
        
        return kernel_copy

    def tune(self, test: KernelTest, tune_params: Dict[str, List[Any]], restrictions: Optional[List[str]] = None, tune_kernel_kwargs: Optional[Dict[str, Any]] = None) -> TuneResult:
        """Tunes the kernel with various parameter configurations to find optimal performance.
        
        This method runs the kernel with different parameter configurations specified in
        ``tune_params``, evaluates each configuration using the provided test case, and returns
        the best performing configuration.
        
        Args:
            test (KernelTest): A KernelTest instance containing input data and expected output to 
                validate the kernel's correctness during tuning.
            tune_params (Dict[str, List[Any]]): A dictionary mapping parameter names to lists of possible values
                to be explored during the tuning process.
            restrictions (Optional[List[str]]): Optional list of restriction strings that define relationships
                between parameters to constrain the search space. Examples include "block_size_x>=tile_size"
                or "block_size_y==block_size_x". If None, no restrictions are applied.
            tune_kernel_kwargs (Optional[Dict[str, Any]]): Optional additional keyword arguments to pass to the
                run_kernel function.
        
        Returns:
            TuneResult: An object containing the best parameter configuration and its
                execution time.
            
        Raises:
            CompileErrorError: If there is an error while compiling the kernel.
            TimeoutError: If the tuning process takes longer than the calculated timeout.
        """
        mp.set_start_method('spawn', force=True)
        kwargs = {'kernel_name': self.kernel_info.name, 
                  'kernel_source': self.code, 
                  'tune_params': tune_params,
                  'arguments': list(test.input_data),
                  'problem_size': test.size,
                  'answer': list(test.expected_output),
                  'restrictions': restrictions,
                  'device': self.kernel_info.device,}
        
        # Check that tune_kernel_kwargs don't override critical parameters
        if tune_kernel_kwargs is not None:
            conflicting_keys = set(kwargs.keys()).intersection(tune_kernel_kwargs.keys())
            assert not conflicting_keys, (
                f"Custom tune_kernel_kwargs cannot override protected parameters: {conflicting_keys}. "
                f"These parameters are set automatically based on the test configuration."
            )
            
            # Add custom tune_kernel keyword arguments
            kwargs.update(tune_kernel_kwargs)

        logger.info(f"Tuning a kernel ```{self.code}``` with params {tune_params}")
        tune_results = self._tune_kernel_process(kwargs)
        logger.info(f"Kernel output matches expected output")
        return tune_results


    def test(self, test: KernelTest, tune_params: Dict[str, Union[int, float]], run_kernel_kwargs: Optional[Dict[str, Any]] = None):
        """Tests the kernel with specific parameter values against a test case.
    
        This method runs the kernel once with the specified parameter configuration and
        verifies that the output matches the expected output from the test case.
        
        Args:
            test (KernelTest): A KernelTest instance containing input data and expected output for
                validating the kernel's correctness.
            tune_params (Dict[str, Union[int, float]]): A dictionary mapping parameter names to their values for this
                specific test run.
            run_kernel_kwargs (Optional[Dict[str, Any]]): Optional additional keyword arguments to pass to the
                run_kernel function.

        
        Raises:
            FailedTestsError: If the kernel's output doesn't match the expected output.
            CompileErrorError: If there is an error while compiling the kernel.
            TimeoutError: If the kernel execution takes longer than the calculated timeout.
        """
        mp.set_start_method('spawn', force=True)
        kwargs = {'kernel_name': self.kernel_info.name, 
                  'kernel_source': self.code, 
                  'problem_size': test.size,
                  'arguments': list(test.input_data), 
                  'params': tune_params,
                  'device': self.kernel_info.device,}
        
        # Check that run_kernel_kwargs don't override critical parameters
        if run_kernel_kwargs is not None:
            conflicting_keys = set(kwargs.keys()).intersection(run_kernel_kwargs.keys())
            assert not conflicting_keys, (
                f"Custom run_kernel_kwargs cannot override protected parameters: {conflicting_keys}. "
                f"These parameters are set automatically based on the test configuration."
            )
            
            # Add custom run_kernel keyword arguments
            kwargs.update(run_kernel_kwargs)
        
        logger.info(f"Testing a kernel ```{self.code}``` with params {tune_params}")
        kernel_output = self._run_kernel_process(kwargs)

        if not self._compare_outputs(kernel_output, test.expected_output):
            raise FailedTestsError()
        logger.info(f"Kernel output matches expected output")
        
    
    def _compare_outputs(self, output: List[TestInputType], expected_output: List[Optional[TestInputType]]) -> bool:
        if len(expected_output) > len(output):
            return False
        
        for i in range(len(expected_output)):
            expected = expected_output[i]
            actual = output[i]
            
            if expected is not None:
                if not np.array_equal(actual, expected):
                    return False
        return True
    
    def _run_kernel_process(self, kwargs: Dict[str, Any]) -> List[TestInputType]:
        q = mp.Queue()
        p = mp.Process(target=self._wrapper, args=(q, run_kernel), kwargs=kwargs)
        p.start()
        #TODO: this is a bottleneck due to the size of the kernel_output. investigate if this can be done in a better way
        try:
            #TODO: this call is a bottleneck. investigate if this can be done in a better way
            success, kernel_output = q.get(timeout=self._calc_timeout(kwargs['params']))
            #if success: kernel_output is a list of numpy arrays
            #otherwise it is a string with the error message
            if success:
                return kernel_output
            else:
                raise CompileErrorError(str(kernel_output))
        except Empty:
            logger.info("Timeout reached, no output from process")
            raise TimeoutError("Timeout reached, no output from process")
        except Exception as e:
            logger.error("Exception in process")
            logger.error(str(e))
            raise CompileErrorError(str(e))
        finally:
            self._cleanup_process(p, q)



    def _tune_kernel_process(self, kwargs: Dict[str, Any]) -> TuneResult:
        q = mp.Queue()
        p = mp.Process(target=self._wrapper, args=(q, tune_kernel), kwargs=kwargs)
        p.start()
        try:
            success, tuner_output = q.get(timeout=self._calc_timeout(kwargs['tune_params']))
            if success:
                _, environment = tuner_output
                best_config = environment['best_config']
                best_time = best_config['time']
                best_params = self._extract_best_params(best_config, kwargs['tune_params'])
                output = TuneResult(best_tune_params=best_params, time=best_time)
            else:
                raise CompileErrorError(str(tuner_output))
        except Empty: #timeout
            logger.info("Timeout reached, no output from process")
            raise TimeoutError("Timeout reached, no output from process")
        except Exception as e: #any other exception
            logger.error("Exception in process")
            logger.error(str(e))
            raise CompileErrorError(str(e))
        finally:
            self._cleanup_process(p, q)
        return output

    def _cleanup_process(self, process: mp.Process, queue: mp.Queue):
        hanging_process_timeout = 10.0
        process.join(timeout=hanging_process_timeout)

        if process.is_alive():
            logger.warning(f"Process {process.pid} is still alive. Terminating.")
            process.terminate()
            # Wait a short period for the process to terminate
            process.join(timeout=hanging_process_timeout) 
            #if still alive after terminate, kill the process
            if process.is_alive():
                logger.warning(f"Process {process.pid} is still alive. Killing.")
                process.kill()
        
        # Close the queue to prevent resource leaks
        queue.close()

        # Clear PyCUDA context caches if PyCUDA is available
        if pycuda_available:
            try:
                pycuda.tools.clear_context_caches() # type: ignore
                logger.debug("Cleared PyCUDA context caches.")
            except Exception as e:
                logger.warning(f"Could not clear PyCUDA context caches: {e}")

    def _extract_best_params(self, best_config: Dict[str, Any], tune_params: Dict[str, Any]) -> Dict[str, Any]:
        best_params:Dict[str, Any] = {}
        for key in tune_params.keys():
            best_params[key] = best_config[key]
        return best_params
    
    @staticmethod
    def _wrapper(q: mp.Queue, f: Callable, **kwargs):
        try:
            output = f(**kwargs)
            result = (True, output)
        except Exception as e:
            print("Exeption that was caught in the process:", str(e))
            result = (False, str(e))
        q.put(result)

    
    def __repr__(self) -> str:
        return f"TunableKernel(name={self.kernel_info.name}, code={self.code}, args={self.kernel_info.args}, best_tune_params={self.best_tune_params}, best_time={self.best_time}, description={self.kernel_info.description})"
    
    def _calc_timeout(self, tune_params: Dict[str, Any]) -> int:
        timeout = self.kernel_info.time_per_test
        num_combinations = 1
        #for each parameter in tune_params multiply by the number of values in the list
        for key in tune_params.keys():
            param_value = tune_params[key]
            if isinstance(param_value, (list, np.ndarray)):
                count = len(param_value)
                if count > 0:
                    num_combinations *= count
            # If it's a single value dict for test(), num_combinations remains 1
        # Scale timeout by the number of combinations, ensure minimum timeout
        timeout = max(int(timeout * num_combinations), self.kernel_info.time_per_test) # Ensure at least base time
        # Add a small buffer time
        timeout += 10 # Add 10 seconds buffer
        return timeout