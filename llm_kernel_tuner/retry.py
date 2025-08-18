from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from typing import Dict, Type, Callable, Any, Optional, Union
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from dataclasses import dataclass
from llm_kernel_tuner.prompts import retry_prompts
from langchain_core.messages import HumanMessage, AIMessage
from pydantic_core._pydantic_core import ValidationError

logger = get_logger(__name__)

@dataclass
class RetryPolicy:
    """Defines a retry policy for a retry wrapper.

    Args:
        max_retries (int, optional): The maximum number of retries to attempt.
        handlers (Optional[Dict[Type[Exception], Callable[[Any, Exception], Any]]]): A dictionary of exception types to handler functions. If an exception is raised during a retry, the handler function will be called with the current state.
        default_handler (Optional[Callable[[Any, Exception], Any]]): A default handler function to use if no specific handler is provided for an exception.

    Note:
        Either ``handlers`` or ``default_handler`` must be provided.

    See Also:
        See :ref:`Retry Policy <retry_policy>` for usage."""
    max_retries: int = 3
    handlers: Optional[Dict[Type[Exception], Callable[[Any, Exception], Any]]] = None
    default_handler: Optional[Callable[[Any, Exception], Any]] = None
    

    def __post_init__(self):
        self.handlers = self.handlers or {}
        if not (self.default_handler or self.handlers):
            raise ValueError("At least one handler must be provided")
        
    def __repr__(self):
        return f"RetryPolicy(max_retries={self.max_retries})"

def create_retry_wrapper(
    what_to_rerty: Union[Callable[..., Any], CompiledStateGraph],
    policy: RetryPolicy,
    state_type: Type[Any] = dict
) -> CompiledStateGraph:
    """Creates a retry wrapper around a graph or a function using the specified policy.
    Will catch any exception made by the function or the graph wrapped and execute function defined in the policy.

    Example usage:
    
    .. code-block:: python

        ...
        graph = graph_builder.compile()

        def value_error_handler(retry_state: State, error: Exception) -> State:
            print("ValueError occurred:", error)
            print(f"Current value: {retry_state['value']}")
            retry_state["value"] += 1
            return retry_state

        retry_policy: RetryPolicy = RetryPolicy(max_retries=7, handlers={ValueError: value_error_handler})
        wrapped_graph = create_retry_wrapper(graph, retry_policy)
    
    The example above will create a retry policy that retries up to 7 times and will call the `value_error_handler` function
    if a `ValueError` is raised during a retry.


    Args:
        what_to_rerty (Callable[[Any], Any], CompiledStateGraph): The graph or a function to wrap
        policy (RetryPolicy): The retry policy to use
        state_type (Type[Any], optional): Type of the state used for the graph

    Returns:
        CompiledStateGraph: Graph or function wrapped with the retry policy
    """

    def retry_node(state: Any) -> Any:
        assert policy.max_retries > 0, "max_retries must be greater than 0"
        assert policy.handlers or policy.default_handler, "At least one handler must be provided"
        attempt = 0
        current_state = state

        while attempt < policy.max_retries:
            try:
                logger.debug(f"Try number {attempt+1}")
                if isinstance(what_to_rerty, CompiledStateGraph):
                    return what_to_rerty.invoke(current_state)
                else: # Callable
                    return what_to_rerty(current_state)
            except Exception as error:
                logger.info("function/subgraph failed with the following error: (type): %s (text): %s", type(error), str(error))
                attempt += 1
                if attempt >= policy.max_retries:
                    # the caller needs to resolve the issue 
                    # because as a callee we don't know what the intet of the wrapped function is 
                    # and thus don't know what the apropriate action for next state is 
                    # otherwise we might end up in an infinite loop because the state didnt change
                    if 'previous_node_error' in current_state:
                        current_state['previous_node_error'] = error
                        return current_state
                    else: 
                        raise error

                handler = policy.handlers.get(type(error), policy.default_handler) if policy.handlers else policy.default_handler
                if handler is None:
                    raise error
                
                try:
                    current_state = handler(current_state, error)
                except Exception as handler_error:
                    # This means there is an error in the handler
                    raise RuntimeError(f"Error handler failed: {str(handler_error)}") from handler_error
                

    # Create a minimal graph with just the retry node
    wrapper = StateGraph(state_type)
    wrapper.add_node("retry", retry_node)
    wrapper.add_edge(START, "retry")
    
    return wrapper.compile()


class WrongArgumentsError(Exception):
    pass

class FailedTestsError(Exception):
    pass

class NoCodeError(Exception):
    pass

class CompileErrorError(Exception):
    pass

class RestrictionCheckError(Exception):
    pass

def wrong_arguments_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    state["messages"].append(HumanMessage(retry_prompts.wrong_tune_params_prompt))
    return state

def failed_tests_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    state["messages"].append(HumanMessage(retry_prompts.failed_tests_prompt))
    return state

def no_code_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    state["messages"].append(HumanMessage(retry_prompts.no_code_prompt))
    return state

def compile_error_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    error_string = str(error)
    formatted_prompt = retry_prompts.compile_error_prompt.format(compiler_error=error_string)

    state["messages"].append(HumanMessage(formatted_prompt))
    return state

def timeout_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    state["messages"].append(HumanMessage(retry_prompts.timeout_prompt))
    return state

def wrong_structure_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"
    assert type(error) == ValidationError, "error must be of type ValidationError"

    errors = error.errors()

    llm_response = errors[0]["input"]
    logger.info(f"LLM response for wrong structure: {llm_response}")

    state["messages"].append(AIMessage(llm_response))
    state["messages"].append(HumanMessage(retry_prompts.wrong_stucture_prompt))
    return state

def restriction_check_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    state["messages"].append(HumanMessage(retry_prompts.invalid_problem_size_prompt))
    return state    



def default_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    error_string = str(error)
    formatted_prompt = retry_prompts.default_error_prompt.format(error=error_string)

    state["messages"].append(HumanMessage(formatted_prompt))
    return state

default_tuner_retry_policy = RetryPolicy(
    max_retries=3,
    handlers={
        WrongArgumentsError: wrong_arguments_handler,
        FailedTestsError: failed_tests_handler,
        NoCodeError: no_code_handler,
        CompileErrorError: compile_error_handler,
        TimeoutError: timeout_handler,
        ValidationError: wrong_structure_handler,
        RestrictionCheckError: restriction_check_handler,
    },
    default_handler=default_handler)





class InvalidTest(Exception):
    pass

class TestTooShort(Exception):
    def __init__(self, current_time, max_time, min_time):
        super().__init__(f"Test too short: {current_time}ms (min: {min_time}ms)")
        self.current_time = current_time
        self.max_time = max_time
        self.min_time = min_time

class TestTooLong(Exception):
    def __init__(self, current_time, max_time, min_time):
        super().__init__(f"Test too long: {current_time}ms (max: {max_time}ms)")
        self.current_time = current_time
        self.max_time = max_time
        self.min_time = min_time

class CodeError(Exception):
    pass

class SharedMemorySizeExceededError(Exception):
    """Exception raised when the requested shared memory size exceeds the maximum allowed size."""
    def __init__(self, data_size: int, max_data_size: int):
        super().__init__(f"Requested shared memory size ({data_size} bytes) exceeds the maximum allowed size ({max_data_size} bytes).")
        self.data_size = data_size
        self.max_data_size = max_data_size



def invalid_tests_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    error_string = str(error)
    if error_string != "":
        formatted_prompt = retry_prompts.invalid_test_generated_with_error_prompt.format(error=error_string)
        state["messages"].append(HumanMessage(formatted_prompt))
    else:
        state["messages"].append(HumanMessage(retry_prompts.invalid_test_generated_prompt))

    return state

def test_too_short_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"
    assert isinstance(error, TestTooShort), "exception has the wrong type"
    
    formatted_prompt = retry_prompts.test_too_short_prompt.format(
        current_exec_time=error.current_time,
        max_exec_time=error.max_time,
        min_exec_time=error.min_time,
    )

    state["messages"].append(HumanMessage(formatted_prompt))
    
    return state

def test_too_long_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"
    assert isinstance(error, TestTooLong), "exception has the wrong type"
    
    formatted_prompt = retry_prompts.test_too_long_prompt.format(
        current_exec_time=error.current_time,
        max_exec_time=error.max_time,
        min_exec_time=error.min_time,
    )

    state["messages"].append(HumanMessage(formatted_prompt))
    
    return state

def syntax_error_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"
    error_string = str(error)

    formatted_prompt = retry_prompts.syntax_error_prompt.format(syntax_error=error_string)

    state["messages"].append(HumanMessage(formatted_prompt))
    return state

def default_tester_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    error_string = str(error)
    formatted_prompt = retry_prompts.default_tester_error_prompt.format(error=error_string)

    state["messages"].append(HumanMessage(formatted_prompt))
    return state

def data_size_too_large_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"
    assert isinstance(error, SharedMemorySizeExceededError), "exception has the wrong type"
    
    formatted_prompt = retry_prompts.data_size_too_large_prompt.format(
        data_size=error.data_size,
        max_data_size=error.max_data_size,
    )

    state["messages"].append(HumanMessage(formatted_prompt))
    
    return state


default_tester_retry_policy = RetryPolicy(
    max_retries=10, #tests are very important so give a bit more retries to run generate
    handlers={
        NoCodeError: no_code_handler,
        InvalidTest: invalid_tests_handler,
        ValidationError: wrong_structure_handler,
        TestTooShort: test_too_short_handler,
        TestTooLong: test_too_long_handler,
        SyntaxError: syntax_error_handler,
        SharedMemorySizeExceededError: data_size_too_large_handler,
    },
    default_handler=default_tester_handler)






class InvalidProblemSize(Exception):
    pass

class InvalidOutputVariables(Exception):
    pass


def invalid_problem_size_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    state["messages"].append(HumanMessage(retry_prompts.invalid_problem_size_prompt))

    return state

def invalid_output_variables_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    state["messages"].append(HumanMessage(retry_prompts.invalid_output_variables_prompt))

    return state

def default_tuner_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    assert "messages" in state, "messages key must be present in state"

    state["messages"].append(HumanMessage(retry_prompts.default_transformer_error_prompt))

    return state
    

default_transformer_retry_policy = RetryPolicy(
    max_retries=3,
    handlers={
        InvalidProblemSize: invalid_problem_size_handler,
        InvalidOutputVariables: invalid_output_variables_handler,
    },
    default_handler=default_tuner_handler)