from llm_kernel_tuner.tuning_strategies.base_tuning_strategy import BaseTuningStrategy, _RETRY_POLICY_DEFAULT_SENTINEL
from llm_kernel_tuner.tuning_state import State
from llm_kernel_tuner.tunable_kernel import TunableKernel
from llm_kernel_tuner.retry import create_retry_wrapper, RetryPolicy, WrongArgumentsError, NoCodeError
from llm_kernel_tuner.prompts import one_prompt_strategy_prompts
from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from llm_kernel_tuner.structured_output import get_structured_llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, START, StateGraph
from typing import Optional, List, Dict, Any, TypedDict, Union
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class OnePromptStrategyState(State, TypedDict):
    candidate_kernel: Optional[TunableKernel]

class OnePromptTuningStrategy(BaseTuningStrategy):
    def __init__(self, retry_policy: Any = _RETRY_POLICY_DEFAULT_SENTINEL):
        super().__init__(retry_policy)
        self.llm: Optional[BaseChatModel] = None

    def create_graph(self, llm: BaseChatModel) -> CompiledStateGraph: 
        self.llm = llm

        graph_builder = StateGraph(OnePromptStrategyState)
        graph_builder.add_node("ask_optimization", self._prompt)
        graph_builder.add_node("test_kernel", self._test_kernel)

        graph_builder.add_edge(START, "ask_optimization")
        graph_builder.add_edge("ask_optimization", "test_kernel")

        original_graph = graph_builder.compile()
        if self.retry_policy is None:
            return original_graph
        retry_graph = create_retry_wrapper(original_graph, self.retry_policy, OnePromptStrategyState)
        return retry_graph
    

    def _prompt(self, state: OnePromptStrategyState) -> OnePromptStrategyState:
        assert(self.llm is not None), "LLM is not set"

        kernel = state["kernel"]

        if len(state["messages"]) == 0:
            new_messages = self._create_messages(kernel)
            state["messages"].extend(new_messages)

        class TunedKernelOutput(BaseModel):
            code: str = Field(description="The tuned kernel code")
            tunable_parameters: List[str] = Field(description="Tunable parameters for the kernel")

        logger.info(f"Invoking LLM to optimize kernel")
        structured_llm = get_structured_llm(self.llm, TunedKernelOutput)
        answer = structured_llm.invoke(state["messages"])

        if isinstance(answer, TunedKernelOutput): # check if answer is of the correctly formatted
            logger.info(f"LLM response: \n```{answer.code}```")
            state["messages"].append(AIMessage(answer.model_dump_json())) #add llm response to messages
            tuned_kernel_code = self._extract_and_sanitize_kernel(answer.code)
            if tuned_kernel_code: #check if code is returned

                tunable_parameters = {}
                if answer.tunable_parameters:
                    tuned_kernel_code, tunable_parameters = self._fix_tunable_parameters(answer.code, answer.tunable_parameters)
                    tuned_kernel_code = self._extract_and_sanitize_kernel(tuned_kernel_code)
                    if tunable_parameters:
                        state["curr_tune_params"] = tunable_parameters

                if tuned_kernel_code:
                    new_kernel = kernel.copy()
                    new_kernel.code = tuned_kernel_code
                    
                    state["candidate_kernel"] = new_kernel
                    return state
            else:
                logger.info(f"No code returned")
                raise NoCodeError("Could not extract code from llm")
    
        raise NoCodeError("LLM didnt return structured code")


    def _create_messages(self, kernel:TunableKernel) -> List[BaseMessage]:
        prompt_template = one_prompt_strategy_prompts.user_prompt
        user_prompt = prompt_template.format(kernel_string=kernel.code)
        messages = [
            SystemMessage(one_prompt_strategy_prompts.system_prompt),
            HumanMessage(user_prompt),
        ]
        return messages
    
    def _fix_tunable_parameters(self, kernel: str, tune_params: List[str]) -> tuple[str, Dict[str, List[Any]]]:
        assert(self.llm is not None), "LLM is not set"
        
        tune_params_str = "["+(", ".join(tune_params))+"]"


        fix_params_prompt = one_prompt_strategy_prompts.fix_params_prompt.format(kernel_string=kernel, tunable_parameters=tune_params_str)

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

        return kernel, {}
    
    def _test_kernel(self, state: OnePromptStrategyState) -> OnePromptStrategyState:
        candidate_kernel = state["candidate_kernel"]
        if candidate_kernel is None:
            return state

        curr_tune_params = state["curr_tune_params"]
        tests = state['tests']
        curr_kernel = state["kernel"]

        restrictions = self._ask_restrictions(curr_kernel.code, curr_tune_params)

        tune_result = self._run_tests(candidate_kernel, curr_tune_params, tests, restrictions)
        logger.info(f"Kernel passed all tests")
        
        # Update candidate kernel with the test results
        candidate_kernel.best_time = tune_result.time
        
        # Check if the new kernel should be accepted based on performance threshold
        if tune_result.time is not None and self._should_accept_kernel(curr_kernel, tune_result.time):
            logger.info(f"New kernel is faster than the old kernel")
            
            # Record the successful optimization step
            step_description = "One-prompt kernel optimization"
            self._record_successful_step(
                step_description=step_description,
                old_kernel=curr_kernel,
                new_kernel=candidate_kernel,
                tune_params=curr_tune_params,
                restrictions=restrictions,
                best_params=tune_result.best_tune_params,
                state=state
            )
            
            state["kernel"] = candidate_kernel
            state['best_params'] = tune_result.best_tune_params
            return state
        
        logger.info(f"New kernel is slower than the old kernel")
        return state