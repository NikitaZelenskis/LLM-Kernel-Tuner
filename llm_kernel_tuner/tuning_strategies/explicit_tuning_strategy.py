from llm_kernel_tuner.tuning_strategies.base_tuning_strategy import BaseTuningStrategy, _RETRY_POLICY_DEFAULT_SENTINEL
from llm_kernel_tuner.tuning_strategies.tuning_step import TuningStep
from llm_kernel_tuner.tunable_kernel import TunableKernel
from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from llm_kernel_tuner.retry import create_retry_wrapper, RetryPolicy, WrongArgumentsError, NoCodeError
from llm_kernel_tuner.prompts import explicit_strategy_prompts
from llm_kernel_tuner.structured_output import get_structured_llm
from typing import Optional, List, Dict, Any, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, Field
from llm_kernel_tuner.tuning_state import State
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, START, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel

logger = get_logger(__name__)


explicit_tuning_steps: List[TuningStep] = [
    TuningStep(id="block_size_x", 
                prompt_template=explicit_strategy_prompts.block_size_x,
                tune_params={"block_size_x": [32, 64, 128, 256, 512, 1024]},
    ),
    TuningStep(id="block_size_y", 
                prompt_template=explicit_strategy_prompts.block_size_y,
                tune_params={"block_size_y": [1, 2, 4, 8, 16, 32]},
    ),
    TuningStep(id="process_2elem_x", 
                prompt_template=explicit_strategy_prompts.process2elem_x,
                depends_on=["block_size_x"],
    ),
    TuningStep(
        id="tunable_nr_elem_x",
        prompt_template=explicit_strategy_prompts.tunable_nr_elem_x,
        tune_params={"work_per_thread_x": [2, 4, 8]},
        depends_on=["process_2elem_x"],
    ),
    TuningStep(
        id="tile_stride_x",
        prompt_template=explicit_strategy_prompts.tile_stride_x,
        tune_params={"work_stride_x": [0, 1]},
        depends_on=["tunable_nr_elem_x"],
    ),
    TuningStep(
        id="process_2elem_y",
        prompt_template=explicit_strategy_prompts.process2elem_y,
        depends_on=["block_size_y"],
    ),
    TuningStep(
        id="tunable_nr_elem_y",
        prompt_template=explicit_strategy_prompts.tunable_nr_elem_y,
        tune_params={"work_per_thread_y": [2, 4, 8]},
        depends_on=["process_2elem_y"],
    ),
    TuningStep(
        id="tile_stride_y",
        prompt_template=explicit_strategy_prompts.tile_stride_y,
        tune_params={"work_stride_y": [0, 1]},
        depends_on=["tunable_nr_elem_y"],
    ),
    TuningStep(
        id="shared_memory_input",
        prompt_template=explicit_strategy_prompts.shared_memory_input,
        tune_params={"use_shared_mem": [0, 1]},
    ),
    TuningStep(
        id="shared_memory_output",
        prompt_template=explicit_strategy_prompts.shared_memory_output,
        tune_params={"output_shared_mem": [0, 1]},
    ),
    TuningStep(
        id="memory_prefetching",
        prompt_template=explicit_strategy_prompts.memory_prefetching,
        depends_on=["shared_memory_input"],
    ),
    TuningStep(
        id="multi_dimensional_tiling_x",
        prompt_template=explicit_strategy_prompts.multi_dimensional_tiling_x,
        tune_params={"tile_size_x": [16, 32, 64]},
        depends_on=["block_size_x"],
    ),
    TuningStep(
        id="multi_dimensional_tiling_y",
        prompt_template=explicit_strategy_prompts.multi_dimensional_tiling_y,
        tune_params={"tile_size_y": [16, 32, 64]},
        depends_on=["multi_dimensional_tiling_x", "block_size_y"],
    ),
    TuningStep(
        id="loop_unrolling",
        prompt_template=explicit_strategy_prompts.loop_unrolling,
        tune_params={"loop_unroll_factor": [2, 4, 8]},
    ),
    TuningStep(
        id="instruction_parallelism",
        prompt_template=explicit_strategy_prompts.instruction_parallelism,
    ),
]




class ExplicitStrategyState(State, TypedDict):
    tuning_steps: List[TuningStep]
    candidate_kernel: Optional[TunableKernel]
    tuning_step_necessary: bool
    completed_tuning_steps: Dict[str, bool]
    

class ExplicitTuningStrategy(BaseTuningStrategy):
    """A strategy that applies a sequence of explicit tuning steps to a kernel.
    
    This tuning strategy follows a predefined sequence of tuning steps, each applying
    a specific optimization technique to the kernel. Each step can have tunable parameters,
    dependencies on previous steps, and can be conditionally applied based on an evaluation.
    
    Args:
        tuning_steps: Optional[List[TuningStep]]: A list of tuning steps to apply.
            If None, uses the default explicit_tuning_steps.
    
    Example:
        .. code-block:: python

            from llm_kernel_tuner import TunableKernel, LLMKernelTransformer
            from llm_kernel_tuner.tuning_strategies import ExplicitTuningStrategy, TuningStep
            from langchain_core.prompts import PromptTemplate
            
            # Define custom tuning steps
            my_tuning_steps = [
                TuningStep(
                    id="shared_memory_tiling",
                    prompt_template=PromptTemplate.from_template(...),
                    tune_params={"tile_size": [16, 32, 64]}
                ),
                TuningStep(
                    id="loop_unrolling",
                    prompt_template=PromptTemplate.from_template(...),
                    tune_params={"unroll_factor": [2, 4, 8]},
                    depends_on=["shared_memory_tiling"]
                )
            ]
            
            # Create the strategy with custom steps
            strategy = ExplicitTuningStrategy(tuning_steps=my_tuning_steps)
            
            # Use the strategy with a kernel transformer
            kernel_transformer = LLMKernelTransformer(
                kernel="...",
                tuning_strategy=strategy
            )
        
    See Also:
        :ref:`explicit_tuning_strategy` for more information on how to use this strategy.
        """
    def __init__(self, tuning_steps: Optional[List[TuningStep]] = None, retry_policy: Any = _RETRY_POLICY_DEFAULT_SENTINEL):
        super().__init__(retry_policy)
        self.llm: Optional[BaseChatModel] = None
        self.tuning_steps = explicit_tuning_steps if tuning_steps is None else tuning_steps

    def create_graph(self, llm: BaseChatModel) -> CompiledStateGraph:
        self.llm = llm

        graph_builder = StateGraph(ExplicitStrategyState)
        graph_builder.add_node("init_new_state", self._init_new_state)
        graph_builder.add_node("is_tuning_step_necessary", self._is_tuning_step_necessary)
        graph_builder.add_node("transform_kernel", self._safe_transform_kernel)
        graph_builder.add_node("next_step", self._next_step)

        graph_builder.add_edge(START, "init_new_state")
        graph_builder.add_edge("init_new_state", "is_tuning_step_necessary")
        graph_builder.add_conditional_edges(
            "is_tuning_step_necessary",
            self._is_tuning_step_necessary_decision,
            {
                "yes": "transform_kernel",
                "no": "next_step"
            }
        )
        graph_builder.add_edge("transform_kernel", "is_tuning_step_necessary")
        graph_builder.add_conditional_edges(
            "next_step",
            self._should_continue_decision,
            {
                "continue": "is_tuning_step_necessary",
                "end": END,
            },
        )

        return graph_builder.compile()
    
    def _safe_transform_kernel(self, state: ExplicitStrategyState) -> ExplicitStrategyState:
        transform_kernel_subgraph = self._creat_transform_subgraph(self.retry_policy)
        try:
            return transform_kernel_subgraph.invoke(state) #type: ignore
        except Exception as e:
            logger.error(f"Tuning step '{state['tuning_steps'][0].id}' failed after all retries: {e}")
            state['tuning_steps'] = state['tuning_steps'][1:]
            state["messages"] = []
            return state

    def _init_new_state(self, state: ExplicitStrategyState) -> ExplicitStrategyState:
        state["tuning_steps"] = self.tuning_steps
        state["completed_tuning_steps"] = {}
        state["messages"] = []
        return state
    
    def _creat_transform_subgraph(self, retry_policy: Optional[RetryPolicy]) -> CompiledStateGraph:
        """
        Wrap llm invocation and testing in retry for fault tolerancy
        """
        subgraph_builder = StateGraph(ExplicitStrategyState)

        subgraph_builder.add_node("init_kernel_creation", self._init_kernel_creation)
        subgraph_builder.add_node("invoke_llm", self._invoke_llm)
        subgraph_builder.add_node("test_candidate_kernel", self._test_candidate_kernel)
        
        subgraph_builder.add_edge(START, "init_kernel_creation")
        subgraph_builder.add_edge("init_kernel_creation", "invoke_llm")
        subgraph_builder.add_edge("invoke_llm", "test_candidate_kernel")
        subgraph = subgraph_builder.compile()
        if retry_policy is None:
            return subgraph
        retry_llm_subgraph = create_retry_wrapper(subgraph, retry_policy, ExplicitStrategyState)
        return retry_llm_subgraph

    def _init_kernel_creation(self,
        state:ExplicitStrategyState
    ) -> ExplicitStrategyState:
        state["candidate_kernel"] = None
        return state


    def _invoke_llm(
        self,
        state:ExplicitStrategyState
    ) -> ExplicitStrategyState:
        assert(self.llm is not None), "LLM is not set"

        kernel = state["kernel"]
        # first tuning step is always the current tuning step
        tuning_step = state['tuning_steps'][0]

        if len(state["messages"]) == 0:
            new_messages = self._create_messages(kernel, tuning_step)
            state["messages"].extend(new_messages)
            
        class TunedKernelOutput(BaseModel):
            code: str = Field(description="The tuned kernel code")
        structured_llm = get_structured_llm(self.llm, TunedKernelOutput)
        answer = structured_llm.invoke(state["messages"])

        if isinstance(answer, TunedKernelOutput): # check if answer is of the correctly formatted
            logger.info(f"LLM response {answer.code}")
            tuned_kernel_code = self._extract_and_sanitize_kernel(answer.code)
            if tuned_kernel_code: #check if code is returned
                state["messages"].append(AIMessage(content=answer.code)) #add llm response to messages
                new_kernel = kernel.copy()
                new_kernel.code = tuned_kernel_code

                state["candidate_kernel"] = new_kernel
                return state
            else:
                logger.info(f"No code returned for tuning step {tuning_step.id}")
                raise NoCodeError("Could not extract code from llm")

        raise NoCodeError("LLM didnt return structured code") 


    def _test_candidate_kernel(
            self,
            state: ExplicitStrategyState
        ) -> ExplicitStrategyState:
        kernel = state["candidate_kernel"]
        if kernel is None:
            raise NoCodeError("Could not extract kernel")

        curr_tune_params = state["curr_tune_params"] | state['tuning_steps'][0].tune_params
        current_tuning_step = state['tuning_steps'][0]

        restrictions = self._ask_restrictions(kernel.code, curr_tune_params)
        #if tests fail they will raise an exception
        tune_result = self._run_tests(kernel, curr_tune_params, state['tests'], restrictions)
        
        logger.info(f"Kernel passed all tests")
        if tune_result.time is not None and self._should_accept_kernel(kernel, tune_result.time):
            # Calculate improvement percentage for logging
            if kernel.best_time is not None:
                improvement_percentage = self._calculate_improvement_percentage(kernel.best_time, tune_result.time)
                threshold = kernel.kernel_info.performance_threshold
                logger.info(f"Kernel accepted: improvement {improvement_percentage:.2f}% meets threshold {threshold}% (old: {kernel.best_time:.6f}s, new: {tune_result.time:.6f}s)")
            else:
                logger.info(f"First kernel accepted with execution time: {tune_result.time:.6f}s")
            
            # Record successful optimization step
            old_kernel = state["kernel"]
            kernel.best_time = tune_result.time
            step_description = self._generate_step_description(current_tuning_step, state['completed_tuning_steps'])
            self._record_successful_step(
                step_description=step_description,
                old_kernel=old_kernel,
                new_kernel=kernel,
                tune_params=curr_tune_params,
                restrictions=restrictions,
                best_params=tune_result.best_tune_params,
                state=state
            )
            
            logger.info(f"New best kernel has been chosen, kernel code: ```{kernel.code}``` with execution time: {tune_result.time}")
            state['best_params'] = tune_result.best_tune_params
            state['curr_tune_params'] = curr_tune_params
            state["kernel"] = kernel
        elif tune_result.time is not None:
            # Kernel was rejected due to insufficient improvement
            improvement_percentage = self._calculate_improvement_percentage(kernel.best_time, tune_result.time)
            threshold = kernel.kernel_info.performance_threshold
            logger.info(f"Kernel rejected: improvement {improvement_percentage:.2f}% does not meet threshold {threshold}% (Current time: {kernel.best_time})")

        completed_step_id = state['tuning_steps'][0].id
        state['completed_tuning_steps'][completed_step_id] = True
        state["candidate_kernel"] = None
        state['tuning_steps'] = [step for step in state['tuning_steps'] if step.id != completed_step_id] # remove completed tuning step

        return state

    def _next_step(self, state: ExplicitStrategyState) -> ExplicitStrategyState:
        state["messages"] = [] # reset messages for next tuning step
        state['tuning_steps'] = state['tuning_steps'][1:]
        return state
    
    def _should_continue_decision(self, state: ExplicitStrategyState) -> str:
        # Check if there are any remaining tuning steps whose dependencies are met
        for step in state["tuning_steps"]:
            if all(dep in state["completed_tuning_steps"] for dep in step.depends_on):
                return "continue"
        return "end"
    
    def _get_next_tuning_step(self, state: ExplicitStrategyState) -> Optional[TuningStep]:
        for step in state["tuning_steps"]:
            if all(dep in state["completed_tuning_steps"] for dep in step.depends_on):
                return step
        return None

    def _is_tuning_step_necessary(self, state: ExplicitStrategyState) -> ExplicitStrategyState:
        """
        Evaluates whether a tuning step makes sense to implement for a given kernel.
        """
        assert self.llm is not None, "LLM is not set"
        
        next_tuning_step = self._get_next_tuning_step(state)

        if next_tuning_step is None:
            state["tuning_step_necessary"] = False
            return state

        kernel = state["kernel"]

        if next_tuning_step.skip_evaluation:
            state["tuning_step_necessary"] = True
            return state

        class EvaluationOutput(BaseModel):
            should_run: bool = Field(description="Whether the tuning step should run")

        prompt = explicit_strategy_prompts.step_evaluation_prompt.format(
            kernel_string=kernel.code, optimization_technique=next_tuning_step.prompt_template
        )

        messages:List[BaseMessage] = [
            HumanMessage(content=prompt),
        ]
        structured_llm = get_structured_llm(self.llm, EvaluationOutput)
        answer = structured_llm.invoke(messages)

        if isinstance(answer, EvaluationOutput):
            logger.info("Is tuning step necessary? LLM response: " + str(answer.should_run))
            state["tuning_step_necessary"] = answer.should_run
            return state
        state["tuning_step_necessary"] = True
        return state
    
    def _is_tuning_step_necessary_decision(self, state: ExplicitStrategyState) -> str:
        if state["tuning_step_necessary"]:
            return "yes"
        else:
            return "no"
        
    def _create_messages(self, kernel: TunableKernel, tuning_step: TuningStep) -> List[BaseMessage]:
        prompt_template = tuning_step.prompt_template
        user_prompt = prompt_template.format(kernel_string=kernel.code)
        messages = [
            SystemMessage(explicit_strategy_prompts.system_prompt),
            HumanMessage(user_prompt),
        ]
        return messages

    def _wrap_code(self, code: str) -> str:
        return f"```CUDA\n{code}\n```"

    def _generate_step_description(self, tuning_step: TuningStep, completed_steps: Dict[str, bool]) -> str:
        """Generate a human-readable description for a tuning step.
        
        This method creates a descriptive string for the optimization step that includes
        the step ID, a brief description of the optimization technique, and information
        about dependencies if applicable.
        
        Args:
            tuning_step: The TuningStep being executed
            completed_steps: Dictionary tracking which steps have been completed
            
        Returns:
            A human-readable description of the optimization step
        """
        # Create base description from step ID
        step_name = tuning_step.id.replace('_', ' ').title()
        
        # Add dependency information if this step has dependencies
        dependency_info = ""
        if tuning_step.depends_on:
            completed_deps = [dep for dep in tuning_step.depends_on if completed_steps.get(dep, False)]
            if completed_deps:
                dep_names = [dep.replace('_', ' ').title() for dep in completed_deps]
                if len(dep_names) == 1:
                    dependency_info = f" (building on {dep_names[0]})"
                else:
                    dependency_info = f" (building on {', '.join(dep_names[:-1])} and {dep_names[-1]})"
        
        # Create comprehensive description
        description = f"{step_name} Optimization{dependency_info}"
        
        return description