from llm_kernel_tuner.tunable_kernel import TunableKernel, TunableKernelInfo
from llm_kernel_tuner.kernel_test import KernelTest
from llm_kernel_tuner.tuning_strategies import BaseTuningStrategy, AutonomousTuningStrategy
from llm_kernel_tuner.prompts import transformer_prompts
from llm_kernel_tuner.testing_strategies.naive_llm_tester import NaiveLLMTester
from llm_kernel_tuner.testing_strategies.base_testing_strategy import BaseTestingStrategy
from llm_kernel_tuner.tuning_state import State
from llm_kernel_tuner.performance_tracker import PerformanceTracker
from llm_kernel_tuner.retry import RetryPolicy, NoCodeError, InvalidProblemSize, InvalidOutputVariables, default_transformer_retry_policy, create_retry_wrapper
from llm_kernel_tuner.structured_output import StructuredOutputType, get_structured_llm
from llm_kernel_tuner.thinking_stripper import ThinkingStripperWrapper
from typing import List, Tuple, Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
import json
import clang.cindex as cl
import os


logger = get_logger(__name__)


# embedded in try block to be able to generate documentation
# and run tests without pycuda installed
try:
    import pycuda.driver as drv
    import pycuda.autoinit
    pycuda_available = True
except ImportError:
    drv = None
    pycuda_available = False


if 'LIBCLANG_PATH' in os.environ:
    cl.Config.set_library_file(os.environ['LIBCLANG_PATH'])
    # cl.Config.set_library_file('/usr/lib/llvm-14/lib/libclang.so')


class LLMKernelTransformer:
    """
    The main class that orchestrates the process of analyzing, transforming, 
    and tuning a given compute kernel using LLMs and predefined strategies.

    It sets up a workflow involving kernel analysis (description, problem size, outputs),
    test generation, test validation, and finally kernel tuning based on the 
    provided strategies and retry policies.

    Args:
        kernel (str): Kernel device code that will be tuned 
        llm (BaseChatModel, optional): The language model instance to use for LLM interactions. 
            Defaults to ChatOpenAI(model="gpt-5").
        tuning_strategy (BaseTuningStrategy, optional): The strategy for tuning the kernel's parameters. 
            Defaults to AutonomousTuningStrategy().
        tests (List[KernelTest], optional): A list of initial tests to validate the kernel's correctness. 
            Defaults to an empty list.
        testing_strategies (List[BaseTestingStrategy]): Strategies for generating tests. 
            Defaults to List[NaiveLLMTester()].
        transformer_retry_policy (RetryPolicy, optional): The retry policy for the kernel analysis (get_kernel_info) subgraph. 
            Defaults to `default_transformer_retry_policy`.
        device (int, optional): The CUDA device ID to use. Defaults to 0.
        clang_args (List[str], optional): Additional arguments that will be provided to clang when parsing the kernel code
        cuda_gpu_arch (Optional[str], optional): Pre-determined CUDA GPU architecture (e.g., "sm_86"). If provided, skips PyCUDA detection. Defaults to None.
        time_per_test (int): The base time limit used for calculating timeouts for kernel tests or tuning runs.
            The time value in seconds.

            For ``tune`` operations:
            The timeout is calculated as ``max(time_per_test * num_combinations, time_per_test) + buffer``,
            where ``num_combinations`` is the product of the number of values for each parameter in
            ``tune_params``, and ``buffer`` is a fixed additional time (currently 10 seconds).
            This ensures the timeout scales with the search space size but is never less than
            ``time_per_test`` plus the buffer.
            
            For ``test`` operations:
            The timeout is calculated as ``time_per_test + buffer``, as there is only one
            parameter combination being tested (``num_combinations`` is effectively 1).
        strip_thinking_output (bool, optional): Whether to strip thinking/reasoning sections from LLM responses. 
            When enabled, the LLM will be wrapped with ThinkingStripperWrapper to remove specified 
            thinking patterns from the output. Defaults to False.
        thinking_pattern (str, optional): Regular expression pattern to match thinking sections 
            that should be stripped from LLM responses. Only used when ``strip_thinking_output`` is True. 
            If not provided, defaults to ``r"<think>.*?</think>\s*"`` which matches content within 
            ``<think>`` tags.
        structured_output_type (StructuredOutputType, optional): The type of structured output format 
            to use for LLM interactions. Defaults to ``StructuredOutputType.DEFAULT``.
        performance_threshold (float, optional): Minimum performance improvement threshold as a percentage 
            required for accepting a new kernel version. This prevents accepting kernels with marginal 
            improvements that may be due to measurement noise or system variability.
            
            The improvement percentage is calculated using the formula:
            ``((old_time - new_time) / old_time) * 100``
            
            A new kernel is accepted only if the calculated improvement percentage is greater than 
            or equal to the threshold value.
            
            Examples:
                - ``performance_threshold=0.5``: Requires at least 0.5% improvement (default)
                - ``performance_threshold=1.0``: Requires at least 1.0% improvement (more conservative)
                - ``performance_threshold=0.0``: Accepts any improvement, however small
            
            Usage examples:
                ```python
                # Default threshold (0.5%)
                transformer = LLMKernelTransformer(kernel_code)
                
                # Conservative threshold (2% improvement required)
                transformer = LLMKernelTransformer(kernel_code, performance_threshold=2.0)
                
                # Accept any improvement
                transformer = LLMKernelTransformer(kernel_code, performance_threshold=0.0)
                ```
            
            Defaults to 0.5.
    """
    default_tune_params:Dict[str, Any] = {'block_size_x': [256]}
    _default_cuda_gpu_arch = "sm_70"

    def __init__(self, kernel_code: str, 
                 llm: Optional[BaseChatModel] = None,
                 tuning_strategy: Optional[BaseTuningStrategy] = None,
                 tests: List[KernelTest] = [], 
                 testing_strategies: List[BaseTestingStrategy] = [],
                 transformer_retry_policy: RetryPolicy = default_transformer_retry_policy,
                 device: int = 0,
                 clang_args: List[str] = [], 
                 cuda_gpu_arch: Optional[str] = None,
                 time_per_test: int = 15,
                 strip_thinking_output: bool = False,
                 thinking_pattern: str = r"<think>.*?</think>\s*",
                 structured_output_type: StructuredOutputType = StructuredOutputType.DEFAULT,
                 performance_threshold: float = 0.5,
                 ):
        # Add basic validation assertion for performance_threshold
        assert isinstance(performance_threshold, (int, float)) and performance_threshold >= 0, \
            "performance_threshold must be a non-negative number"
        
        # Add input validation for performance_threshold
        if not isinstance(performance_threshold, (int, float)) or performance_threshold < 0:
            raise ValueError("performance_threshold must be a non-negative number")
        
        self.kernel_code = kernel_code
        self.kernel_info = self._create_kernel_info(kernel_code, device, clang_args, cuda_gpu_arch, time_per_test, performance_threshold)
        self.kernel = TunableKernel(self.kernel_code, self.kernel_info)

        base_llm = llm if llm is not None else ChatOpenAI(model="gpt-5")
        
        if strip_thinking_output:
            # Could maybe be done with a Langchains Runnable (LCEL) in the future? 
            self.llm: BaseChatModel = ThinkingStripperWrapper(llm=base_llm, pattern=thinking_pattern)
        else:
            self.llm: BaseChatModel = base_llm

        self.tests = tests
        self.tuning_strategy: BaseTuningStrategy = tuning_strategy if tuning_strategy else AutonomousTuningStrategy()
        self.testing_strategies: List[BaseTestingStrategy] = testing_strategies if len(testing_strategies) != 0 else [NaiveLLMTester()]
        self.test_generation_subgraph = self._create_test_generation_subgraph(self.testing_strategies)
        self.tuning_graph = self.tuning_strategy.create_graph(self.llm)
        self.transformer_retry_policy = transformer_retry_policy
        self.workflow = self._create_workflow()
        self.max_recursion_limit = 10000 # langchain needs to have a limit on the numer of times recursion is allowed
        self.structured_output_type = structured_output_type
        self.llm.metadata = {"structured_output_type": structured_output_type}

    def _create_kernel_info(self, kernel_code: str, device: int = 0, clang_args: List[str] = [], cuda_gpu_arch: Optional[str] = None, time_per_test: int = 15, performance_threshold: float = 0.5) -> TunableKernelInfo:
        """
        Returns kernel info object with all values that could be filled in statically.
        Fields that cannot be extracted statically are not filled in (e.g. description, problem_size and output_variables)
        These fields will be filled in later by prompting llm in ``_get_kernel_info()``
        """
        kernel_info = TunableKernelInfo()
        kernel_info.device = device
        kernel_info.clang_args = clang_args
        kernel_info.time_per_test = time_per_test
        kernel_info.performance_threshold = performance_threshold
        if cuda_gpu_arch:
            kernel_info.cuda_gpu_arch = cuda_gpu_arch
        elif pycuda_available and drv is not None:
            try:
                dev = drv.Device(kernel_info.device) # type: ignore
                attrs = dev.get_attributes()
                major = attrs[drv.device_attribute.COMPUTE_CAPABILITY_MAJOR] # type: ignore
                minor = attrs[drv.device_attribute.COMPUTE_CAPABILITY_MINOR] # type: ignore
                kernel_info.cuda_gpu_arch = f"sm_{major}{minor}"
                logger.debug(f"Detected CUDA compute capability for device {kernel_info.device}: {kernel_info.cuda_gpu_arch}")
            except drv.Error as e: # type: ignore
                logger.warning(f"PyCUDA error while detecting compute capability for device {kernel_info.device}: {e}. Falling back to {self._default_cuda_gpu_arch}.")
                kernel_info.cuda_gpu_arch = self._default_cuda_gpu_arch # Fallback default
            except Exception as e:
                logger.warning(f"Unexpected error while detecting compute capability for device {kernel_info.device}: {e}. Falling back to {self._default_cuda_gpu_arch}.")
                kernel_info.cuda_gpu_arch = self._default_cuda_gpu_arch # Fallback default
        else:
            logger.warning(f"PyCUDA not available. Falling back to {self._default_cuda_gpu_arch} for CUDA GPU architecture.")
            kernel_info.cuda_gpu_arch = self._default_cuda_gpu_arch # Fallback default if pycuda is not installed

        kernel_info.name, kernel_info.args = self._extract_kernel_info(kernel_code, kernel_info.clang_args, kernel_info.cuda_gpu_arch)
        if not kernel_info.name:
            raise ValueError("Could not extract kernel name from the provided code.")


        return kernel_info

    
    def _extract_kernel_info(self, code: str, additional_clang_args: List[str], cuda_gpu_arch: str) -> Tuple[str, Dict[str, str]]:
        index = cl.Index.create()

        virtual_file_name = 'virtual_cuda_kernel.cu'
        unsaved_files = [(virtual_file_name, code)]
    
        try:
            # Combine default args with additional user-provided args
            clang_args = [f'--cuda-gpu-arch={cuda_gpu_arch}', '-x', 'cuda'] + additional_clang_args
            tu = index.parse(virtual_file_name, args=clang_args, unsaved_files=unsaved_files)

            # Check for compilation errors
            errors = [d for d in tu.diagnostics if d.severity >= cl.Diagnostic.Error] # type: ignore
            if errors:
                error_messages = "\n".join([str(e) for e in errors])
                logger.info(f"Failed to compile/parse :\n{error_messages}")
                raise Exception(f"Failed to compile/parse :\n{error_messages}")
        except Exception as e:
            raise e

        params:Dict[str, str] = {}
        internal_function_name = ""
        for node in tu.cursor.walk_preorder():
            if node.kind == cl.CursorKind.FUNCTION_DECL: # type: ignore
                if node.location.file.name == virtual_file_name:
                    internal_function_name = node.spelling
                    for param in node.get_arguments():
                        params[param.spelling] = param.type.spelling

        return internal_function_name, params


    def _create_workflow(self):
        assert self.tuning_graph is not None, "Tuning strategy must be provided"
        assert self.test_generation_subgraph is not None, "Testing strategies must be provided"

        workflow = StateGraph(State)
        get_kernel_info_subgraph = self._get_kernel_info()
        workflow.add_node("get_kernel_info", get_kernel_info_subgraph)
        workflow.add_node("test_generation_subgraph", self.test_generation_subgraph)
        workflow.add_node("calc_initial_exec_time", self._calc_initial_exec_time)
        workflow.add_node("tuning_subgraph", self.tuning_graph)

        workflow.add_edge(START, "get_kernel_info")
        workflow.add_edge("get_kernel_info", "test_generation_subgraph")
        workflow.add_edge("test_generation_subgraph", "calc_initial_exec_time")
        workflow.add_edge("calc_initial_exec_time", "tuning_subgraph")
        workflow.add_edge("tuning_subgraph", END)
        
        return workflow
    
    def _calc_initial_exec_time(self, state: State) -> State:
        kernel = state["kernel"]
        tests = state["tests"]
        tune_params = state["curr_tune_params"]

        tune_result = kernel.tune(tests[0], tune_params)
        state["best_params"] = tune_result.best_tune_params
        state["kernel"].best_time = tune_result.time

        # Set baseline time in performance tracker for total improvement calculation
        performance_tracker = state["performance_tracker"]
        performance_tracker.set_baseline_time(tune_result.time)

        logger.info(f"Initial kernel time: {tune_result.time}, with following tuning parameters: {tune_result.best_tune_params}")

        return state

    def _create_test_generation_subgraph(self, testing_strategies: List[BaseTestingStrategy]) -> CompiledStateGraph:
        assert len(testing_strategies) != 0, "There needs to be at least one testing strategy in place" 

        testing_subgraph = StateGraph(State)
        
        # create all testing_strategy graphs and store their names
        node_names = []
        for i, testing_strategy in enumerate(testing_strategies):
            testing_strat_node_name = "testing_strategy_"+str(i)
            node_names.append(testing_strat_node_name)
            testing_strategy_node = testing_strategy.create_graph(self.llm)
            testing_subgraph.add_node(testing_strat_node_name, testing_strategy_node)
            
        # Connect START to the first strategy
        testing_subgraph.add_edge(START, node_names[0])

        # Connect strategies sequentially
        for i in range(len(node_names) - 1):
            testing_subgraph.add_edge(node_names[i], node_names[i+1])
        
        # Connect the last strategy to END
        testing_subgraph.add_edge(node_names[-1], END)
            

        return testing_subgraph.compile()

    
    def make_kernel_tunable(self) -> Tuple[TunableKernel, Dict[str, Any]]:
        """
        | Transforms the kernel to make it tunable.
        
        Returns:
            Tuple[TunableKernel, Dict[str, Any]]: The transformed kernel as :class:`TunableKernel` and the best tuning parameters as a dictionary
        """
        graph = self.workflow.compile()
        final_state = graph.invoke({
            "kernel": self.kernel,
            "best_params": None,
            "tests": self.tests,
            "llm": self.llm,
            "messages": [],
            "curr_tune_params": self.default_tune_params,
            "performance_tracker": PerformanceTracker(),
        }, {"recursion_limit": self.max_recursion_limit})
        logger.info("Done tuning kernel.")
        logger.info(f'Final best performaing kernel: `\n{final_state["kernel"]}\n` with following tunable parameters: `\n{final_state["best_params"]}\n`')
        
        # Display performance overview after tuning completion
        performance_tracker = final_state["performance_tracker"]
        overview = performance_tracker.generate_overview()
        print(overview)
        logger.info("Performance overview displayed.")
        
        return final_state["kernel"], final_state["best_params"]

    def _get_kernel_info(self) -> CompiledStateGraph:
        
        graph_builder = StateGraph(State)

        get_problem_size_with_retry = create_retry_wrapper(self._get_problem_size, self.transformer_retry_policy)
        get_outputs_with_retry = create_retry_wrapper(self._get_outputs, self.transformer_retry_policy)
        
        graph_builder.add_node("get_kernel_description", self._get_kernel_description)
        graph_builder.add_node("get_problem_size", get_problem_size_with_retry)
        graph_builder.add_node("get_outputs_with_retry", get_outputs_with_retry)


        graph_builder.add_edge(START, "get_kernel_description")
        graph_builder.add_edge("get_kernel_description", "get_problem_size")
        graph_builder.add_edge("get_problem_size", "get_outputs_with_retry")

        return graph_builder.compile()
    

    def _get_kernel_description(self, state: State) -> State:
        assert(self.llm is not None), "LLM is not set"

        kernel = state["kernel"].copy()# make a copy to not overwrite original kernel given by the user
        prompt = transformer_prompts.kernel_description_prompt.format(kernel_string=kernel.code)
        messages = [
            HumanMessage(prompt)
        ]
        logger.info(f"Invoking llm to get kernel description for: ```{kernel.code}```")
        answer = self.llm.invoke(messages)
        answerMessage = self._extract_message(answer)
        logger.info(f"LLMs description of the kernel: {answerMessage}")
        kernel.kernel_info.description = answerMessage

        state["kernel"] = kernel
        state["messages"] = []
        return state 

    
    def _get_problem_size(self, state: State) -> State:
        assert self.llm is not None, "LLM must be set"

        class ProblemSizeOutput(BaseModel):
            problem_size: List[str] = Field(description="The problem size")

        kernel = state['kernel']
        
        #if not empty, it means we are in the retry and there are error messages in state['messages']
        if len(state['messages']) == 0:
            state['messages'] = [
                SystemMessage(transformer_prompts.problem_size_prompt),
                HumanMessage("```CUDA\n"+kernel.code+"\n```")
            ]

        structured_llm = get_structured_llm(self.llm, ProblemSizeOutput)
        answer = structured_llm.invoke(state['messages'])

        if isinstance(answer, ProblemSizeOutput):
            logger.info(f"LLM response for problem size: {answer.problem_size}")
            state['messages'] = []
            state['kernel'].kernel_info.problem_size = answer.problem_size
            return state

        state['messages'].append(AIMessage(''))
        raise InvalidProblemSize("LLM did not provide structured output")


    
    def _get_outputs(self, state: State) -> State:
        """Performs a single attempt to ask the LLM for kernel output variables.

        This method interacts with the configured LLM, providing the kernel code
        and asking it to identify the output variables. 

        Args:
            state: An OutputsState dictionary containing:
                - 'kernel': The TunableKernel object.
                - 'messages': A list of messages exchanged so far with the LLM
                  (used for retries with context).

        Raises:
            AssertionError: If the LLM has not been set before calling this method.
            InvalidOutputVariables: If the LLM responds with an empty list of
                output variables.
            NoCodeError: If the LLM fails to provide a structured output matching
                the expected format.
        """
        assert self.llm is not None, "LLM must be set"

        class OutputVariables(BaseModel):
            output_variables: List[str] = Field(description="output variables of the kernel")

        kernel = state['kernel']
        
        #if not empty, it means we are in the retry and there are error messages in state['messages']
        if len(state['messages']) == 0:
            state['messages'] = [
                SystemMessage(transformer_prompts.extract_output_var_prompt),
                HumanMessage("```CUDA\n"+kernel.code+"\n```")
            ]

        structured_llm = get_structured_llm(self.llm, OutputVariables) 
        answer = structured_llm.invoke(state['messages'])

        if isinstance(answer, OutputVariables):
            logger.info(f"LLM response for outputs: {answer.output_variables}")
            state['messages'].append(AIMessage("output_variables: "+json.dumps(answer.output_variables)))
            if not answer.output_variables:
                raise InvalidOutputVariables("Could not extract output variables from LLM answer")
            state['kernel'].kernel_info.output_variables = answer.output_variables
            state['messages'] = []
            return state

        state['messages'].append(AIMessage(''))
        raise NoCodeError("LLM did not provide structured output")

    def _extract_message(self, answer: BaseMessage) -> str:
        answerMessage:Optional[str] = None
        if isinstance(answer, BaseMessage):
            if isinstance(answer.content, str):
                answerMessage = answer.content
        if isinstance(answer, str):#llma_cpp returns a string instead of a BaseMessage
            answerMessage = answer
        
        if answerMessage is None:
            raise ValueError("LLM returned a unexpected message")
        
        return answerMessage

    def add_test(self, test: KernelTest):
        """
        Add test to the test suite. This test will be used to test the correctness of the kernel while it is being tuned.

        Args:
            test (KernelTest): test to be added to the testsuite.
        """
        self.tests.append(test)


