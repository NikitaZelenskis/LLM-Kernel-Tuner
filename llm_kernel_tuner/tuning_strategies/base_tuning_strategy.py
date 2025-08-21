from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from llm_kernel_tuner.tunable_kernel import TuneResult, TunableKernel
from llm_kernel_tuner.kernel_test import KernelTest
from llm_kernel_tuner.retry import RetryPolicy, default_tuner_retry_policy, create_retry_wrapper, RestrictionCheckError
from llm_kernel_tuner.prompts import tuning_strategy_prompts
from llm_kernel_tuner.performance_tracker import PerformanceTracker, PerformanceStep
from typing import Optional, List, Dict, Any, overload, TypedDict, TYPE_CHECKING
from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
import re
import json
from datetime import datetime
from llm_kernel_tuner.structured_output import get_structured_llm

if TYPE_CHECKING:
    from llm_kernel_tuner.tuning_state import State

logger = get_logger(__name__)

_RETRY_POLICY_DEFAULT_SENTINEL = object()  # Sentinel object for default retry_policy

class AskRestrictionState(TypedDict):
    kernel_code: str
    tune_params: Dict[str, List[Any]]
    restrictions: List[str]
    messages: List[BaseMessage]

class BaseTuningStrategy(ABC):
    """Base class for kernel tuning strategies.
    
    This abstract class defines the interface for tuning strategies that optimize
    kernel parameters. Implementations of this class should tune the kernel, do the testing of the kernel
    and evaluation themselves. 

    Note:
        | Subclasses must implement the :func:`create_graph <BaseTuningStrategy.create_graph>` method.
        | :class:`llm_kernel_tuner.tuning_state.State` will be passed to this graph.

    See Also:
        :ref:`custom_tuning_strategy` for example usage.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, retry_policy: Optional[RetryPolicy]) -> None: ...

    def __init__(self, retry_policy: Any = _RETRY_POLICY_DEFAULT_SENTINEL):
        self._test_cache: Dict[str, TuneResult] = {}
        self.llm: Optional[BaseChatModel] = None
        if retry_policy is _RETRY_POLICY_DEFAULT_SENTINEL:
            self.retry_policy: Optional[RetryPolicy] = default_tuner_retry_policy
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


    def _generate_cache_key(self, kernel_code: str, tune_params: Dict[str, List[Any]]) -> str:
        """Generates a unique cache key based on kernel code and tuning parameters."""
        # Sort the tune_params dictionary by key to ensure consistent ordering
        sorted_params = json.dumps(tune_params, sort_keys=True)
        return f"{kernel_code}::{sorted_params}"



    def _run_tests(
        self, kernel: TunableKernel, tune_params: Dict[str, List[Any]], tests: List[KernelTest], restrictions: Optional[List[str]] = None
    ) -> TuneResult:
        """Tunes the kernel with the first test and evaluates correctness with remaining tests.
        
        Args:
            kernel (TunableKernel): The tunable kernel to optimize.
            tune_params (Dict[str, List[Any]]): Dictionary mapping parameter names to lists of possible values.
            tests (List[KernelTest]): List of kernel tests to execute.
            restrictions (List[str]): List of restriction strings that define relationships between parameters to constrain the search space (e.g., "block_size_x>=tile_size").
        
        Returns:
            TuneResult: Test result containing the tuning performance metrics.
            
        Note:
            Only the first test is used for parameter tuning to save computational resources.
            The remaining tests validate the correctness of the kernel.
            Results are cached based on kernel code and tuning parameters.
        """
        cache_key = self._generate_cache_key(kernel.code, tune_params)
        if cache_key in self._test_cache:
            logger.debug(f"Cache hit for kernel code and tune_params. Returning cached result.")
            return self._test_cache[cache_key]

        tune_result = TuneResult()
        best_params: Dict[str, Any] = {}
        # First test is being tuned, the rest are being run with the best tuning parameters
        if len(tests) > 0:
            tune_result = kernel.tune(tests[0], tune_params, restrictions)
            best_params = tune_result.best_tune_params
        for test in tests[1:]:
            kernel.test(test, best_params)
        
        # Store the result in cache before returning
        self._test_cache[cache_key] = tune_result
        return tune_result

    def _extract_and_sanitize_kernel(
        self, answer_prompt: str
    ) -> Optional[str]:
        """Extracts and sanitizes kernel code from the model's response.
        
        This method processes an LLM response to extract only the kernel code.
        It first removes preprocessor directives and then extracts code that appears between triple backticks.

        Note:
            The extracted and sanitized kernel code, or None if no code could be extracted.
           
        Args:
            answer_prompt (str): The text response potentially containing kernel code.
            
        Returns:
            The extracted and sanitized kernel code, or None if no code could be extracted.
            
        Example:
            Given an LLM response like:
            
            .. code-block:: text

                Here's an optimized CUDA kernel:
                
                ```cuda
                #define BLOCK_SIZE 256
                
                __global__ void matrixMul(float* A, float* B, float* C, int width) {
                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    
                    if (row < width && col < width) {
                        float sum = 0.0f;
                        for (int i = 0; i < width; i++) {
                            sum += A[row * width + i] * B[i * width + col];
                        }
                        C[row * width + col] = sum;
                    }
                }
                ```
            
            The method will return:
            
            .. code-block:: text

                __global__ void matrixMul(float* A, float* B, float* C, int width) {
                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                    
                    if (row < width && col < width) {
                        float sum = 0.0f;
                        for (int i = 0; i < width; i++) {
                            sum += A[row * width + i] * B[i * width + col];
                        }
                        C[row * width + col] = sum;
                    }
                }
            
        """
        code = self._remove_preprocessor_directives(answer_prompt)

        # Extract code from triple backticks if present
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            # If no backticks are found, assume the whole sanitized response is code
            code = code.strip()

        if code:
            return code
        return None

    def _remove_preprocessor_directives(self, code: str) -> str:
        """Removes preprocessor directives from kernel code.
    
        This method filters out lines starting with "#define" from the provided code.
        
        Args:
            code (str): The kernel code string from which to remove preprocessor directives.
            
        Returns:
            str: The kernel code with "#define" directives removed.
            
        Example:
            Given code:
            
            .. code-block:: text
            
                #define BLOCK_SIZE 256
                __global__ void myKernel() {
                    // kernel implementation
                }
                
            This method returns:
            
            .. code-block:: text
            
                __global__ void myKernel() {
                    // kernel implementation
                }
        """
        lines = code.split("\n")
        # cleaned_lines = [line for line in lines if not line.strip().startswith(('#define', '#if', '#endif', '#ifndef'))]
        cleaned_lines = [line for line in lines if not line.strip().startswith("#define")]
        return "\n".join(cleaned_lines)
    

    def _ask_restrictions(self, kernel_code: str, tune_params: Dict[str, List[Any]]) -> List[str]:
        """Ask the LLM to generate parameter restrictions for kernel tuning.

        This method queries the language model to determine appropriate restrictions
        between tuning parameters based on the kernel code and available parameters.
        The restrictions help constrain the parameter search space during optimization.

        Args:
            kernel_code (str): The CUDA kernel source code to analyze.
            tune_params (Dict[str, List[Any]]): Dictionary mapping parameter names 
                to lists of possible values that can be tuned.

        Returns:
            List[str]: A list of restriction strings that define relationships 
                between parameters (e.g., "block_size_x==block_size_y" or 
                "tile_size<=block_size_x").

        Raises:
            RestrictionCheckError: If the LLM generates restrictions referencing 
                parameters not present in tune_params.

        Note:
            This method uses the configured retry policy if available. If restrictions
            contain invalid parameter names, the operation will be retried according
            to the retry policy configuration.

        Example:
            Given a matrix multiplication kernel with parameters:
            
            .. code-block:: python
            
                tune_params = {
                    "block_size_x": [16, 32, 64],
                    "block_size_y": [16, 32, 64],
                    "tile_size": [2, 4, 8]
                }
                
            The method might return:
            
            .. code-block:: python
            
                ["block_size_x>=tile_size", "block_size_y>=tile_size"]

        """
        new_retry_state: AskRestrictionState = {
            "kernel_code": kernel_code,
            "tune_params": tune_params,
            "restrictions": [],
            "messages": [],
        } 

        if self.retry_policy is None:
            new_retry_state = self._ask_restrictions_inner(new_retry_state)
            return new_retry_state["restrictions"]
        else:
            retry_invoke = create_retry_wrapper(self._ask_restrictions_inner, self.retry_policy, AskRestrictionState)
            new_retry_state: AskRestrictionState = retry_invoke.invoke(new_retry_state) # type: ignore
            return new_retry_state["restrictions"]
    
    def _ask_restrictions_inner(self, state: AskRestrictionState) -> AskRestrictionState:
        assert(self.llm is not None), "LLM is not set"

        kernel_code = state["kernel_code"]
        tune_params = state["tune_params"]

        if len(state["messages"]) == 0:
            get_restrictions_prompt = tuning_strategy_prompts.get_restrictions_prompt.format(kernel_string=kernel_code, tune_params=json.dumps(tune_params))
            state['messages'] = [
                HumanMessage(get_restrictions_prompt)
            ]

        class GetRestrictions(BaseModel):
            restrictions: List[str] = Field(description="List of restrctions")

        logger.info("Asking llm for restrictions")
        structured_llm = get_structured_llm(self.llm, GetRestrictions)
        answer = structured_llm.invoke(state['messages'])
        
        if isinstance(answer, GetRestrictions):
            logger.info(f"LLM response: restrictions: {answer.restrictions}")
            state['messages'].append(AIMessage(answer.model_dump_json()))

            if not self._check_restrictions_validity(answer.restrictions, tune_params):
                raise RestrictionCheckError()
            else:
                state["restrictions"] = answer.restrictions
        return state
    
    def _check_restrictions_validity(self, restrictions: List[str], tune_params: Dict[str, List[Any]]) -> bool:
        """Check if all parameters mentioned in restrictions are present in tune_params.
    
        This method validates that all parameter names found in restriction strings
        are actually available in the tune_params dictionary.
        Args:
            restrictions (List[str]): List of restriction strings like "block_size_y==block_size_x" or "block_size_x==block_size_y*tile_size_y"
            tune_params (Dict[str, List[Any]]): Dictionary mapping parameter names to lists of possible values
            
        Returns:
            True if all parameters in restrictions exist in tune_params, False otherwise.
        """
        available_params = set(tune_params.keys())
        regex_match_variable = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"

        for restriction in restrictions:
            param_names = re.findall(regex_match_variable, restriction)
            
            for param_name in param_names:
                if param_name not in available_params:
                    logger.warning(f"Parameter '{param_name}' in restriction '{restriction}' not found in tune_params")
                    return False
        
        return True

    def _calculate_improvement_percentage(self, old_time: float, new_time: float) -> float:
        """Calculate the performance improvement percentage between old and new execution times.
        
        Args:
            old_time (float): The previous execution time
            new_time (float): The new execution time
            
        Returns:
            float: The improvement percentage using formula: ((old_time - new_time) / old_time) * 100
                  Positive values indicate improvement, negative values indicate regression
                  
        Note:
            Handles edge cases for zero or very small time values by returning 0.0
            when old_time is zero or near-zero to avoid division by zero.
        """
        # Handle edge case where old_time is zero or very small
        if old_time <= 1e-10:  # Very small threshold to handle floating point precision
            logger.debug(f"Edge case: old_time ({old_time}) is zero or near-zero, returning 0.0% improvement")
            return 0.0
        
        # Handle edge case where new_time is negative (invalid)
        if new_time < 0:
            logger.warning(f"Edge case: new_time ({new_time}) is negative, treating as no improvement")
            return -100.0  # Treat as significant regression
            
        improvement_percentage = ((old_time - new_time) / old_time) * 100
        
        # Log when there's no improvement or regression
        if new_time >= old_time:
            logger.debug(f"No improvement: new_time ({new_time}) >= old_time ({old_time}), improvement: {improvement_percentage:.2f}%")
        
        return improvement_percentage

    def _meets_performance_threshold(self, old_time: float, new_time: float, threshold: float) -> bool:
        """Check if the performance improvement meets the specified threshold.
        
        Args:
            old_time (float): The previous execution time
            new_time (float): The new execution time  
            threshold (float): The minimum improvement percentage required
            
        Returns:
            bool: True if the calculated improvement percentage meets or exceeds the threshold,
                  False otherwise
        """
        improvement_percentage = self._calculate_improvement_percentage(old_time, new_time)
        return improvement_percentage >= threshold

    def _should_accept_kernel(self, current_kernel: TunableKernel, new_time: float) -> bool:
        """Determine if a new kernel should be accepted based on performance threshold.
        
        Args:
            current_kernel (TunableKernel): The current best kernel
            new_time (float): The execution time of the new kernel candidate
            
        Returns:
            bool: True if the new kernel should be accepted, False otherwise
            
        Note:
            Always accepts the first kernel (when current_kernel.best_time is None).
            For subsequent kernels, uses threshold comparison to determine acceptance.
        """
        # Always accept the first kernel
        if current_kernel.best_time is None:
            logger.debug("Accepting first kernel (no previous best_time)")
            return True
        
        # Handle edge case where new_time is invalid
        if new_time < 0:
            logger.warning(f"Rejecting kernel with invalid negative execution time: {new_time}")
            return False
            
        # Use threshold comparison for subsequent kernels
        threshold = current_kernel.kernel_info.performance_threshold
        old_time = current_kernel.best_time
        improvement_percentage = self._calculate_improvement_percentage(old_time, new_time)
        
        meets_threshold = self._meets_performance_threshold(old_time, new_time, threshold)
        
        if meets_threshold:
            logger.debug(f"Accepting kernel: improvement {improvement_percentage:.2f}% >= threshold {threshold}%")
        else:
            logger.debug(f"Rejecting kernel: improvement {improvement_percentage:.2f}% < threshold {threshold}%")
            
        return meets_threshold

    def _record_successful_step(self, step_description: str, old_kernel: TunableKernel, 
                              new_kernel: TunableKernel, tune_params: Dict[str, List[Any]], 
                              restrictions: List[str], best_params: Dict[str, Any], 
                              state: 'State') -> None:
        """Record a successful optimization step in the performance tracker.
        
        This method creates a PerformanceStep record and adds it to the performance
        tracker when a kernel optimization step is accepted based on the performance
        threshold. It captures comprehensive information about the optimization.
        
        Args:
            step_description: Human-readable description of the optimization step
            old_kernel: The previous kernel before this optimization step
            new_kernel: The new optimized kernel after this step
            tune_params: The tunable parameters used for this optimization
            restrictions: Parameter restrictions applied during tuning
            best_params: The best parameter values found for this kernel
            state: The tuning state containing the performance tracker
            
        Note:
            This method should only be called when a kernel optimization step
            has been accepted (i.e., meets the performance threshold).
            The performance tracker is initialized by the LLMKernelTransformer.
        """
        
        # Calculate improvement percentage
        old_time = old_kernel.best_time
        new_time = new_kernel.best_time
        
        if new_time is None:
            logger.warning("Cannot record step: new kernel has no execution time")
            return
            
        improvement_percentage = self._calculate_improvement_percentage(
            old_time if old_time is not None else 0.0, 
            new_time
        )
        
        # Create performance step record
        step = PerformanceStep(
            step_description=step_description,
            kernel_code=new_kernel.code,
            old_execution_time=old_time,
            new_execution_time=new_time,
            improvement_percentage=improvement_percentage,
            tunable_parameters=tune_params.copy(),
            restrictions=restrictions.copy(),
            best_tune_params=best_params.copy(),
            timestamp=datetime.now()
        )
        
        # Record the step
        state['performance_tracker'].record_step(step)
        
        logger.info(f"Recorded successful optimization step: {step_description} "
                   f"(improvement: {improvement_percentage:.2f}%)")

