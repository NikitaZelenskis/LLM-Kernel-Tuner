.. _retry_policy:

Retry Policy
============

This guide demonstrates how to build reliable LLM-powered systems that can recover from failures using the retry mechanism in LangGraph.

Overview
--------

When working with Large Language Models (LLMs) for code generation, various failures can occur: generated code might contain syntax errors or fail to meet required optimization standards. 
These issues are particularly common when generating complex code or specialized algorithms.

Instead of implementing complex error handling throughout your application, the :func:`create_retry_wrapper <llm_kernel_tuner.retry.create_retry_wrapper>` mechanism 
provides an elegant solution for automatically handling errors, making corrections, and retrying operations.

Basic Usage
-----------

Let's start with a simple function that compiles code and might fail:

.. code-block:: python

    from typing import Dict, Any, TypedDict
    
    class CompilationError(Exception):
        """Exception raised when code compilation fails."""
        pass
        
    class State(TypedDict):
        code: str
        compilation_attempts: int
    
    def compile_code(state: State) -> State:
        """
        Simulates compiling code which might fail for different reasons.
        """
        print(f"Attempt #{state['compilation_attempts']}: Compiling code...")
        
        if "syntax error" in state["code"].lower():
            raise CompilationError("Syntax error detected in the code")
        
        print("Compilation successful!")
        return state

Error Handling
--------------

Just like in real life, the compilation step might raise an error.
When a function fails, we need a way to recover, make adjustments, and retry the operation.
This is where :func:`create_retry_wrapper <llm_kernel_tuner.retry.create_retry_wrapper>` can help to recover from a failed state.

First, you need to define error handlers like so:

.. code-block:: python

    def compilation_error_handler(retry_state: State, error: Exception) -> State:
        print(f"Compilation Error: {str(error)}")
        
        # Attempt to fix the code if it has syntax errors
        if "syntax error" in str(error).lower():
            print("Attempting to fix syntax error...")
            retry_state["code"] = retry_state["code"].replace("syntax error here", "")
        
        # Increment attempt counter
        retry_state["compilation_attempts"] += 1

        return retry_state

Creating a Retry Policy
-----------------------

We'll define a :func:`RetryPolicy <llm_kernel_tuner.retry.RetryPolicy>` that specifies how many retries to attempt and which handlers to use for different error types:

.. code-block:: python

    from llm_kernel_tuner.retry import RetryPolicy
    retry_policy = RetryPolicy(
        max_retries=5,
        handlers={
            CompilationError: compilation_error_handler
        }
    )

Direct Function Wrapping
------------------------

The simplest way to use the retry mechanism is to wrap a function directly. This approach is perfect when you don't need the complexity of a LangGraph subgraph:

.. code-block:: python

    from llm_kernel_tuner.retry import create_retry_wrapper

    # Wrap the compile_code function with retry logic
    retry_compile_code = create_retry_wrapper(
        compile_code,
        retry_policy
    )

    # Now retry_compile_code can be used in your LangGraph:
    build_graph_builder = StateGraph(State)
    build_graph_builder.add_node("compilation_with_retry", retry_compile_code)
    build_graph_builder.add_edge(START, "compilation_with_retry")
    build_graph_builder.add_edge("compilation_with_retry", END)
    build_graph = build_graph_builder.compile()

Integrating with LangGraph Subgraphs
------------------------------------

For more complex scenarios, you can also wrap entire LangGraph subgraphs:

.. code-block:: python

    from langgraph.graph import StateGraph, START, END
    from llm_kernel_tuner.retry import create_retry_wrapper

    # Create a subgraph for the compilation process
    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node("compile", compile_code)
    subgraph_builder.add_edge(START, "compile")
    subgraph_builder.add_edge("compile", END)
    compilation_subgraph = subgraph_builder.compile()

    # Wrap the compilation subgraph with retry logic
    retry_compilation_subgraph = create_retry_wrapper(
        compilation_subgraph,
        retry_policy
    )

    # Use the wrapped subgraph in a larger workflow
    build_graph_builder = StateGraph(State)
    build_graph_builder.add_node("compilation_with_retry", retry_compilation_subgraph)
    build_graph_builder.add_edge(START, "compilation_with_retry")
    build_graph_builder.add_edge("compilation_with_retry", END)
    build_graph = build_graph_builder.compile()

.. note::
    If one of the nodes in the graph fails the whole graph will be retried from the start.


Testing Our Retry System
------------------------

Let's test our new graph by giving it code with a syntax error:

.. code-block:: python

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

Complete Example
----------------

Here is the full example that demonstrates both approaches to retry handling:

.. code-block:: python

    from typing import Dict, Any, TypedDict
    from langgraph.graph import StateGraph, START, END
    from llm_kernel_tuner.retry import create_retry_wrapper, RetryPolicy

    class CompilationError(Exception):
        """Exception raised when code compilation fails."""
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

    # Option 1: Directly wrap the function
    retry_compile_code = create_retry_wrapper(
        compile_code,
        retry_policy
    )

    # Build a graph using the wrapped function
    direct_graph_builder = StateGraph(State)
    direct_graph_builder.add_node("compilation_with_retry", retry_compile_code)
    direct_graph_builder.add_edge(START, "compilation_with_retry")
    direct_graph_builder.add_edge("compilation_with_retry", END)
    direct_graph = direct_graph_builder.compile()

    # Option 2: Create and wrap a subgraph
    subgraph_builder = StateGraph(State)
    subgraph_builder.add_node("compile", compile_code)
    subgraph_builder.add_edge(START, "compile")
    subgraph_builder.add_edge("compile", END)
    compilation_subgraph = subgraph_builder.compile()

    # Wrap the compilation subgraph with retry logic
    retry_compilation_subgraph = create_retry_wrapper(
        compilation_subgraph,
        retry_policy
    )

    # Build a graph using the wrapped subgraph
    subgraph_based_builder = StateGraph(State)
    subgraph_based_builder.add_node("compilation_with_retry", retry_compilation_subgraph)
    subgraph_based_builder.add_edge(START, "compilation_with_retry")
    subgraph_based_builder.add_edge("compilation_with_retry", END)
    subgraph_based_graph = subgraph_based_builder.compile()

    # Try to use the wrapped graph with problematic code
    initial_state = {
        "code": "function main() { console.log('Hello world'); syntax error here }",
        "compilation_attempts": 0,
    }

    try:
        result = direct_graph.invoke(initial_state)
        print("\nBuild completed successfully!")
        print(f"Final state: {result}")
        print(f"Total compilation attempts: {result['compilation_attempts']}")
    except Exception as e:
        print(f"\nBuild failed after multiple attempts: {e}")