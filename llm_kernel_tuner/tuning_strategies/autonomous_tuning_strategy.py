from llm_kernel_tuner.tuning_strategies.base_tuning_strategy import BaseTuningStrategy, _RETRY_POLICY_DEFAULT_SENTINEL
from llm_kernel_tuner.tunable_kernel import TunableKernel
from llm_kernel_tuner.tuning_state import State
from llm_kernel_tuner.retry import create_retry_wrapper, RetryPolicy, WrongArgumentsError, NoCodeError, default_tuner_retry_policy
from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from llm_kernel_tuner.prompts import autonomous_strategy_prompts
from llm_kernel_tuner.structured_output import get_structured_llm
from typing import Optional, List, Dict, Any, TypedDict, Union
from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import json

logger = get_logger(__name__)

class AutonomousStrategyState(State, TypedDict):
    plan: List[str]
    plan_breakdown_counts: List[int]
    past_steps: List[str]
    last_step_broken_down: bool
    last_step_valid: bool
    replan: bool
    previous_node_error: Optional[Exception]
    replan_count: int


class AutonomousTuningStrategy(BaseTuningStrategy):
    """A fully autonomous strategy for kernel optimization based on the plan-and-solve approach.
    
    This strategy first generates an optimization plan and then executes each step sequentially.
    Steps can be broken down recursively into smaller steps and the strategy can replan if needed.
    The process includes validation of steps, breaking them down if necessary, and executing them
    while ensuring correctness.

    Args:
        retry_policy (RetryPolicy, optional): Retry policy for LLM calls. 
            If not provided, defaults to `default_tuner_retry_policy`.
            Can be set to `None` for no retries.
        breakdown_steps (bool, optional): Whether to enable breakdown of steps. 
            If True, the strategy will recursively breakdown each step into smaller steps.
            Defaults to False.
        with_replanning (bool, optional): Whether to enable replanning. 
            If True, the strategy will re-prompt the LLM for additional steps if needed.
            Defaults to True.
        max_breakdowns (int, optional): The maximum number of breakdown steps allowed per step.
            Because steps are broken down recursively the number of steps will grow exponentially, therefore it is advisable to not set this parameter higher than 2.
            Must be greater than 0 if `breakdown_steps` is enabled. 
            Defaults to 1.
        max_replanning (int, optional): The maximum number of replanning attempts allowed.
            Must be greater than 0 if `with_replanning` is enabled.
            Defaults to 3.

    See Also:
        For a detailed explanation of the strategy workflow and examples, see :ref:`autonomous_tuning_strategy`.
    """
    def __init__(self, breakdown_steps: bool = False, with_replanning: bool = True, max_breakdowns: int = 1, max_replanning: int = 3, retry_policy: Any = _RETRY_POLICY_DEFAULT_SENTINEL):
        super().__init__(retry_policy)
        if breakdown_steps:
            assert max_breakdowns > 0, "max_breakdowns must be greater than 0"
        if with_replanning:
            assert max_replanning > 0, "max_replanning must be greater than 0"
        self.llm: Optional[BaseChatModel] = None
        self.breakdown_steps = breakdown_steps
        self.with_replanning = with_replanning
        self.max_breakdowns = max_breakdowns
        self.max_replanning = max_replanning

    def create_graph(self, llm: BaseChatModel) -> CompiledStateGraph:
        self.llm = llm

        graph_builder = StateGraph(AutonomousStrategyState)
        self._add_init_new_state_node(graph_builder)

        self._add_should_end_node(graph_builder)
        self._add_validation_node(graph_builder)
        if self.breakdown_steps:
            self._add_breakdown_node(graph_builder)

        self._add_agent_node(graph_builder)

        if self.with_replanning:
            self._add_replan_node(graph_builder)


        graph = graph_builder.compile()
        return graph
    
    def _add_init_new_state_node(self, graph: StateGraph):
        assert self.retry_policy is not None, "Retry policy is not set"
        graph.add_node("init_new_state", self._init_new_state)
        graph.add_node("planner", create_retry_wrapper(self._planner, self.retry_policy, AutonomousStrategyState))

        graph.add_edge(START, "init_new_state")
        graph.add_edge("init_new_state", "planner")
        graph.add_edge("planner", "should_end_step")


    def _add_should_end_node(self, graph: StateGraph):
        graph.add_node("should_end_step", self._should_end_step)

        if self.with_replanning:
            graph.add_conditional_edges("should_end_step", self._should_end, ["validate_step", "replan_step"])
        else:
            graph.add_conditional_edges("should_end_step", self._should_end, ["validate_step", END])


    def _add_validation_node(self, graph: StateGraph):
        assert self.retry_policy is not None, "Retry policy is not set"
        graph.add_node("validate_step", create_retry_wrapper(self._validate_step, self.retry_policy, AutonomousStrategyState))

        validation_path_map = ["should_end_step"]
        if self.breakdown_steps:
            validation_path_map.append("breakdown_plan")
        else:
            validation_path_map.append("agent")

        graph.add_conditional_edges("validate_step", self._validate_step_condition , validation_path_map)

    def _add_breakdown_node(self, graph: StateGraph):
        assert self.retry_policy is not None, "Retry policy is not set"
        graph.add_node("breakdown_plan", create_retry_wrapper(self._breakdown_plan, self.retry_policy, AutonomousStrategyState))
        #if step has been broken down then check if the new step needs to be broken down
        #so this creates a loop for breaking down steps
        graph.add_conditional_edges("breakdown_plan", 
                                    self._has_been_broken_down, 
                                    ["validate_step", "agent"])

    def _add_agent_node(self, graph: StateGraph):
        assert self.retry_policy is not None, "Retry policy is not set"
        retry_agent_subgraph = create_retry_wrapper(self._agent, self.retry_policy, AutonomousStrategyState)
        graph.add_node("agent", retry_agent_subgraph)
        graph.add_edge("agent", "should_end_step")

    def _add_replan_node(self, graph: StateGraph):
        assert self.retry_policy is not None, "Retry policy is not set"
        graph.add_node("replan_step", create_retry_wrapper(self._replan_step, self.retry_policy, AutonomousStrategyState))

        if self.with_replanning:
            path_map = ["should_end_step", END]

            graph.add_conditional_edges("replan_step",
                                        self._should_replan,
                                        path_map)

    def _has_been_broken_down(self, state: AutonomousStrategyState) -> str:
        if state["last_step_broken_down"]:
            return "validate_step"
        return "agent"

    def _init_new_state(self, state: AutonomousStrategyState) -> AutonomousStrategyState:
        state["plan"] = []
        state["past_steps"] = []
        state["previous_node_error"] = None
        state["replan_count"] = 0
        return state
    
    def _should_end_step(self, state: AutonomousStrategyState) -> AutonomousStrategyState:
        if state["previous_node_error"]:
            current_step = state["plan"][0]
            state["past_steps"].append(current_step)
            state["plan"] = state["plan"][1:]
            state["plan_breakdown_counts"] = state["plan_breakdown_counts"][1:]
            state["previous_node_error"] = None
        return state
    
    def _should_replan(self, state: AutonomousStrategyState) -> str:
        if state["replan"]:
            return "should_end_step"
        return END
    
    def _should_end(self, state: AutonomousStrategyState) -> str:
        if not state["plan"] or len(state["plan"]) == 0:
            if self.with_replanning:
                return "replan_step"
            else:
                return END
        return "validate_step"
    
    def _validate_step_condition(self, state: AutonomousStrategyState) -> str:
        if state["last_step_valid"]:
            if self.breakdown_steps:
                return "breakdown_plan"
            return "agent"
        return "should_end_step"
    
    def _validate_step(self, state: AutonomousStrategyState) -> AutonomousStrategyState:
        assert(self.llm is not None), "LLM is not set"

        current_step = state["plan"][0]
        validate_step_prompt = autonomous_strategy_prompts.validate_step_prompt.format(optimization_step=current_step)

        #if not empty, it means we are in the retry and there are error messages in state['messages']
        if len(state["messages"]) == 0:
            state['messages'] = [
                HumanMessage(validate_step_prompt)
            ]

        class ValidateStep(BaseModel):
            valid: bool = Field(description="Whether the step is valid or not")
        
        logger.info(f"Validating step: \"{current_step}\"")
        structured_llm = get_structured_llm(self.llm, ValidateStep)
        answer = structured_llm.invoke(validate_step_prompt)
        if isinstance(answer, ValidateStep):
            logger.info(f"LLM answered with validation: {answer}")
            state["last_step_valid"] = answer.valid
            if not answer.valid:
                state["past_steps"].append(current_step)
                state["plan"] = state["plan"][1:]
                state["plan_breakdown_counts"] = state["plan_breakdown_counts"][1:]
            state['messages'] = []
            return state
        
        raise NoCodeError("LLM didnt return structured code")

    def _breakdown_plan(self, state: AutonomousStrategyState) -> AutonomousStrategyState:
        assert(self.llm is not None), "LLM is not set"
        assert(self.breakdown_steps > 0), "Breakdown steps is not set"
        
        kernel = state["kernel"]
        current_step = state["plan"][0]
        current_breakdown_count = state["plan_breakdown_counts"][0]

        if current_breakdown_count >= self.max_breakdowns:
            logger.info(f"Max breakdown count reached for step: \"{current_step}\"")
            state["last_step_broken_down"] = False
            return state



        #if not empty, it means we are in the retry and there are error messages in state['messages']
        if len(state["messages"]) == 0:
            break_down_prompt = autonomous_strategy_prompts.breakdown_step_prompt.format(kernel_string=kernel.code, current_step=current_step)
            state['messages'] = [
                HumanMessage(break_down_prompt)
            ]

        class Breakdown(BaseModel):
            breakdown: bool = Field(description="Whether the step needs to be broken down")
            steps: List[str] = Field(description="Broken down steps that will be used to optimize CUDA kernel, each item in the array is a separate step")

        logger.info(f"Asking for breakdown of step: \"{current_step}\"")
        strucctured_llm = get_structured_llm(self.llm, Breakdown)
        answer = strucctured_llm.invoke(state['messages'])
        if isinstance(answer, Breakdown):
            logger.info(f"LLM answered with breakdown: {answer}")
            state["last_step_broken_down"] = answer.breakdown
            if answer.breakdown:
                # For each new step, assign a breakdown count that is parent's count + 1.
                new_breakdown_count = current_breakdown_count + 1
                new_steps = answer.steps
                new_counts = [new_breakdown_count] * len(new_steps)
                # Prepend the new steps and their counts to the plan.
                state["plan"] = new_steps + state["plan"][1:]
                state["plan_breakdown_counts"] = new_counts + state["plan_breakdown_counts"][1:]
            state["messages"] = []
            return state

        raise NoCodeError("LLM didnt return structured code")


    def _planner(self, state: AutonomousStrategyState) -> AutonomousStrategyState:
        assert(self.llm is not None), "LLM is not set"

        kernel = state["kernel"]

        prompt = autonomous_strategy_prompts.initial_planning_prompt.format(kernel_string=kernel.code)

        #if not empty, it means we are in the retry and there are error messages in state['messages']
        if len(state["messages"]) == 0:
            state['messages'] = [
                HumanMessage(prompt)
            ]

        class Plan(BaseModel):
            steps: List[str] = Field(description="List of steps to be executed, each item in the array is a separate step")

        logger.info("Asking for initial planning")
        strucctured_llm = get_structured_llm(self.llm, Plan)
        answer = strucctured_llm.invoke(state['messages'])
        if isinstance(answer, Plan):
            logger.info(f"LLM answered with plan: \"{answer.steps}\"")
            state["plan"] = answer.steps
            state["plan_breakdown_counts"] = [0] * len(answer.steps)
            state["messages"] = []
            return state
        
        raise NoCodeError("LLM didnt return structured code")

    
    def _agent(self, state: AutonomousStrategyState) -> AutonomousStrategyState:
        assert(self.llm is not None), "LLM is not set"

        kernel = state["kernel"]
        current_step = state["plan"][0]
        curr_tune_params = state["curr_tune_params"]

        agent_prompt = autonomous_strategy_prompts.agent_prompt.format(kernel_string=kernel.code, optimization_technique=current_step, tunable_parameters=curr_tune_params)

        #if not empty, it means we are in the retry and there are error messages in state['messages']
        if len(state["messages"]) == 0:
            state['messages'] = [
                HumanMessage(agent_prompt)
            ]

        class TunedKernelOutput(BaseModel):
            code: str = Field(description="The tuned kernel code")
            tunable_parameters: List[str] = Field(description="Tunable parameters for the kernel")

        logger.info(f"Applying optimization step: \"{current_step}\"")
        structured_llm = get_structured_llm(self.llm, TunedKernelOutput)
        answer = structured_llm.invoke(state['messages'])
        
        if isinstance(answer, TunedKernelOutput):
            logger.info(f"LLM answered with tuned kernel: \"{answer.code}\" and tunable parameters: \"{answer.tunable_parameters}\"")
            state['messages'].append(AIMessage(answer.model_dump_json()))

            tuned_kernel_code = answer.code
            tunable_parameters = curr_tune_params
            if answer.tunable_parameters:
                tuned_kernel_code, tunable_parameters = self._fix_tunable_parameters(answer.code, answer.tunable_parameters, curr_tune_params)
                
            tuned_kernel_code = self._remove_preprocessor_directives(tuned_kernel_code)
            if tuned_kernel_code:
                restrictions = self._ask_restrictions(tuned_kernel_code, tunable_parameters)
                new_kernel = kernel.copy()
                new_kernel.code = tuned_kernel_code
                state = self._test_kernel(new_kernel, tunable_parameters, restrictions, state)
                state["messages"] = []
                return state
            else:
                logger.info(f"Could not extract code from llm")
                raise NoCodeError("Could not extract code from llm")
    
        raise NoCodeError("LLM didnt return structured code")


    def _fix_tunable_parameters(self, kernel: str, new_params: List[str], curr_tune_params: Dict[str, Any]) -> tuple[str, Dict[str, List[Any]]]:
        assert(self.llm is not None), "LLM is not set"
        
        new_params_str = "["+(", ".join(new_params))+"]"
        curr_tune_params_str = json.dumps(curr_tune_params)


        fix_params_prompt = autonomous_strategy_prompts.fix_params_prompt.format(kernel_string=kernel, existing_tunable_parameters=curr_tune_params_str, new_tunable_parameters=new_params_str)

        class TunableParameters(BaseModel):
            name: str = Field(description="The name of the tunable parameter")
            values: List[Union[int, str]] = Field(description="The values of the tunable parameter")

        class FixParams(BaseModel):
            kernel_code: str = Field(description="The kernel code with fixed tunable parameters")
            new_params: List[TunableParameters] = Field(description="The new tunable parameters")

        logger.info("Asking llm to fix tunable parameters")
        structured_llm = get_structured_llm(self.llm, FixParams)
        answer = structured_llm.invoke(fix_params_prompt)
        if isinstance(answer, FixParams):
            logger.info(f"LLM response: kernel_code: {answer.kernel_code}, new_params: {answer.new_params}")
            if answer.new_params:
                new_params_dict:Dict[str, List[Any]] = {}
                for param in answer.new_params:
                    new_params_dict[param.name] = param.values

                if answer.kernel_code:
                    return answer.kernel_code, new_params_dict
                else:
                    return kernel, new_params_dict

        return kernel, curr_tune_params

    


    def _replan_step(self, state: AutonomousStrategyState) -> AutonomousStrategyState:
        assert(self.llm is not None), "LLM is not set"
        if state["replan_count"] >= self.max_replanning:
            logger.info(f"Max replanning count {self.max_replanning} reached")
            state["replan"] = False
            return state
        
        kernel = state["kernel"]

        #if not empty, it means we are in the retry and there are error messages in state['messages']
        if len(state["messages"]) == 0:
            past_steps = state["past_steps"].copy()
            past_steps_str = self._past_steps_to_string(past_steps)
            replan_prompt = autonomous_strategy_prompts.replan_prompt.format(kernel_string=kernel.code, past_steps=past_steps_str)
            state['messages'] = [
                HumanMessage(replan_prompt)
            ]


        class Replan(BaseModel):
            replan: bool = Field(description="Whether to replan or not")
            steps: List[str] = Field(description="New steps to be executed")
        
        logger.info("Asking llm for replan")
        structured_llm = get_structured_llm(self.llm, Replan)
        answer = structured_llm.invoke(state['messages'])
        if isinstance(answer, Replan):
            logger.info(f"LLM response: replan: {answer.replan}, steps: {answer.steps}")
            state["replan"] = answer.replan
            if answer.replan:
                state["replan_count"] += 1
                state["plan"] = answer.steps
                state["plan_breakdown_counts"] = [0] * len(answer.steps)
            state["messages"] = []
            return state
        
        #dead code
        logger.info("LLM did not return structured output")
        state["replan"] = False
        return state

        
    def _past_steps_to_string(self, past_steps: List[str]) -> str:
        return "\n".join([f"- {step}" for step in past_steps])



    def _test_kernel(self, new_kernel: TunableKernel, tune_params: Dict[str, List[Any]], restrictions: List[str], state: AutonomousStrategyState) -> AutonomousStrategyState:
        curr_kernel = state["kernel"]
        current_step = state["plan"][0]
        # if new_kernel.args != curr_kernel.args:
        #     logger.info(f"Kernel does not have the same arguments as the original kernel")
        #     raise WrongArgumentsError(new_kernel.code)
        
        tune_result = self._run_tests(new_kernel, tune_params, state['tests'], restrictions)
        logger.info(f"Kernel passed all tests")
        
        if tune_result.time is not None:
            # Use threshold-based decision logic
            should_accept = self._should_accept_kernel(curr_kernel, tune_result.time)
            
            if should_accept:
                # Calculate improvement percentage for logging
                if curr_kernel.best_time is not None:
                    improvement_percentage = self._calculate_improvement_percentage(curr_kernel.best_time, tune_result.time)
                    threshold = curr_kernel.kernel_info.performance_threshold
                    logger.info(f"Kernel accepted: improvement {improvement_percentage:.2f}% meets threshold {threshold}%. "
                              f"Previous time: {curr_kernel.best_time}, new time: {tune_result.time}")
                else:
                    logger.info(f"First kernel accepted with execution time: {tune_result.time}")
                
                logger.info(f"New best kernel has been chosen, kernel code: ```{new_kernel.code}``` with execution time: {tune_result.time}")
                
                state["kernel"] = new_kernel
                state["kernel"].best_time = tune_result.time
                state['best_params'] = tune_result.best_tune_params
                state["curr_tune_params"] = tune_params
            else:
                # Kernel rejected due to insufficient improvement
                improvement_percentage = self._calculate_improvement_percentage(curr_kernel.best_time, tune_result.time)
                threshold = curr_kernel.kernel_info.performance_threshold
                logger.info(f"Kernel rejected: improvement {improvement_percentage:.2f}% does not meet threshold {threshold}%. "
                          f"Current time: {curr_kernel.best_time}")
        
        state["past_steps"].append(current_step)
        state["plan"] = state["plan"][1:]
        state["plan_breakdown_counts"] = state["plan_breakdown_counts"][1:]

        return state