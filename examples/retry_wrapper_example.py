from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from llm_kernel_tuner.retry import create_retry_wrapper, RetryPolicy

class CompilationError(Exception):
    """Exception raised when code compilation fails."""
    pass

class DependencyError(Exception):
    """Exception raised when a dependency is missing."""
    pass

class State(TypedDict):
    code: str
    compilation_attempts: int

def compile_code(state: State) -> State:
    """
    Simulates compiling code which might fail for different reasons.
    """
    print(f"Attempt #{state['compilation_attempts']}: Compiling code...")
    
    # Simulate different failure scenarios
    if "syntax error" in state["code"].lower():
        raise CompilationError("Syntax error detected in the code")
    
    print("Compilation successful!")
    return state

# Create the subgraph for the compilation process
subgraph_builder = StateGraph(State)
subgraph_builder.add_node("compile", compile_code)
subgraph_builder.add_edge(START, "compile")
subgraph_builder.add_edge("compile", END)
compilation_subgraph = subgraph_builder.compile()

# Define retry handlers for different types of errors
def compilation_error_handler(retry_state: State, error: Exception) -> State:
    print(f"Compilation Error: {str(error)}")
    
    # Attempt to fix the code if it has syntax errors
    if "syntax error" in str(error).lower():
        print("Attempting to fix syntax error...")
        retry_state["code"] = retry_state["code"].replace("syntax error here", "")
    
    # Increment attempt counter
    retry_state["compilation_attempts"] += 1
    
    return retry_state

# Create a retry policy
retry_policy = RetryPolicy(
    max_retries=5,
    handlers={
        CompilationError: compilation_error_handler
    }
)

# Wrap the compilation subgraph with retry logic
retry_compilation_subgraph = create_retry_wrapper(
    compile_code,
    retry_policy
)

build_graph_builder = StateGraph(State)
build_graph_builder.add_node("compilation_with_retry", retry_compilation_subgraph)
build_graph_builder.add_edge(START, "compilation_with_retry")
build_graph_builder.add_edge("compilation_with_retry", END)
build_graph = build_graph_builder.compile()

# Try to use the wrapped graph with problematic code
initial_state = {
    "code": "function main() { console.log('Hello world'); syntax error here }",
    "compilation_attempts": 0,
}

try:
    result = build_graph.invoke(initial_state)
    print("\nBuild completed successfully!")
    print(f"Final state: {result}")
    print(f"Total compilation attempts: {result['compilation_attempts']}")
except Exception as e:
    print(f"\nBuild failed after multiple attempts: {e}")