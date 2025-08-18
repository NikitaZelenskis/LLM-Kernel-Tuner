from llm_kernel_tuner.testing_strategies.base_testing_strategy import BaseTestingStrategy, _RETRY_POLICY_DEFAULT_SENTINEL
from llm_kernel_tuner.prompts import tester_prompts
from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from llm_kernel_tuner.retry import create_retry_wrapper, RetryPolicy, NoCodeError, InvalidTest, default_tester_retry_policy, TestTooShort, TestTooLong
from llm_kernel_tuner.tuning_state import State
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Optional, List, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph.state import CompiledStateGraph
from llm_kernel_tuner.structured_output import get_structured_llm


logger = get_logger(__name__)


class NaiveLLMTester(BaseTestingStrategy):
    def __init__(self, 
                 num_tests: int = 3, 
                 retry_policy: Any = _RETRY_POLICY_DEFAULT_SENTINEL,
                 min_duration_ms: float = 500.0,
                 max_duration_ms: float = 30000.0,
                 max_data_size: int = 5 * 1024**3): #5GB
        super().__init__(retry_policy, max_data_size)

        self.num_tests:int = num_tests
        self.retry_policy: Optional[RetryPolicy] = None
        
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms


    def create_graph(self, llm: BaseChatModel, retry_policy: RetryPolicy = default_tester_retry_policy) -> CompiledStateGraph:
        self.llm = llm
        self.retry_policy = retry_policy

        retry_graph = create_retry_wrapper(self._generate_default_tests, self.retry_policy, State)

        return retry_graph

    def _generate_default_tests(self, state: State) -> State:
        
        self._prepare_messages(state)
        test_code = self._get_test_code_from_llm(state)
        self._generate_and_validate_tests(state, test_code)
        
        logger.info("Successfully generated tests")
        state['messages'] = []
        return state

    def _prepare_messages(self, state: State) -> None:
        """Prepare initial messages if not already set (for retries)."""
        if len(state['messages']) == 0:
            kernel = state['kernel']
            prompt = tester_prompts.test_prompt.format(kernel_string=kernel.code)
            state['messages'] = [
                SystemMessage(tester_prompts.system_prompt),
                HumanMessage(prompt)
            ]

    def _get_test_code_from_llm(self, state: State) -> str:
        """Get test code from LLM using structured output."""
        assert self.llm is not None, "LLM must be set"
        
        class GeneratedTestOutput(BaseModel):
            generated_code: str = Field(description="Test code")

        logger.info("Generating test")
        structured_llm = get_structured_llm(self.llm, GeneratedTestOutput)
        answer = structured_llm.invoke(state['messages'])
        
        if not isinstance(answer, GeneratedTestOutput):
            logger.info("Failed to generate structured output")
            raise NoCodeError("LLM returned empty test code")
        
        logger.info(f"LLM response for generating test: {answer.generated_code}")
        state['messages'].append(AIMessage(answer.generated_code))
        
        test_code = answer.generated_code
        if test_code == "":
            raise NoCodeError("LLM returned empty test code")
        
        return test_code

    def _generate_and_validate_tests(self, state: State, test_code: str) -> None:
        """Generate multiple tests and validate the first one with tuning."""
        kernel = state['kernel']
        
        for i in range(self.num_tests):
            test = self._create_test_from_code(kernel, test_code, state)
            
            # Only tune/time the first test for validation
            if i == 0:
                self._validate_test_timing(test, state)
            
            state['tests'].append(test)

    def _create_test_from_code(self, kernel, test_code: str, state: State):
        """Create a test object from the generated code."""
        try:
            return self.get_test_from_code(
                kernel, 
                test_code, 
                state["curr_tune_params"], 
                self.max_duration_ms/1000
            )
        except TimeoutError:
            raise TestTooLong(
                f"More than {self.max_duration_ms}", 
                self.max_duration_ms, 
                self.min_duration_ms
            )
        except:
            raise  # Re-raise all other exceptions

    def _validate_test_timing(self, test, state: State) -> None:
        """Validate test timing by running a tune operation."""
        try:
            tune_results = state["kernel"].tune(test, state["curr_tune_params"])
            if tune_results.time is not None:
                exec_time_ms = tune_results.time * 1000
                self._check_timing_constraints(exec_time_ms)
        except (TestTooLong, TestTooShort):
            raise  # Re-raise these specific exceptions
        except:
            # Catch all other exceptions and ignore them
            # This is because it is possible to fail the tuning process due to restrictions being None
            pass

    def _check_timing_constraints(self, exec_time_ms: float) -> None:
        """Check if execution time is within acceptable bounds."""
        if exec_time_ms > self.max_duration_ms:
            raise TestTooLong(exec_time_ms, self.max_duration_ms, self.min_duration_ms)
        elif exec_time_ms < self.min_duration_ms:
            raise TestTooShort(exec_time_ms, self.max_duration_ms, self.min_duration_ms)