.. _custom_tuning_strategy:

Custom tuning strategy
======================

To implement your own tuning strategy you will need to extend from :class:`BaseTuningStrategy <llm_kernel_tuner.tuning_strategies.BaseTuningStrategy>`, which is located at ``llm_kernel_tuner.tuning_strategies``.
You will need to implement the following function in your custom strategy ``def create_graph(self, llm: BaseChatModel) -> CompiledStateGraph``.

Here is a example:

.. code-block:: python

    from llm_kernel_tuner.tuning_strategies.base_tuning_strategy import BaseTuningStrategy
    from llm_kernel_tuner.retry import RetryPolicy, default_tuner_retry_policy
    from langgraph.graph.state import CompiledStateGraph
    from llm_kernel_tuner.tuning_state import State
    from langgraph.graph import END, START, StateGraph
    from langchain_core.language_models.chat_models import BaseChatModel
    from typing import Optional


    class MyCustomTuningStrategy(BaseTuningStrategy):
        def __init__(self, retry_policy: Optional[RetryPolicy] = default_tuner_retry_policy):
            super().__init__(retry_policy)
            self.llm: Optional[BaseChatModel] = None

        def create_graph(self, llm: BaseChatModel) -> CompiledStateGraph: 
            self.llm = llm

            graph_builder = StateGraph(State)
            graph_builder.add_node("llm_invocation", self.llm_invocation)
            graph_builder.add_node("test_kernel", self.test_kernel)

            graph_builder.add_edge(START, "llm_invocation")
            graph_builder.add_edge("llm_invocation", "test_kernel")

            retry_graph = create_retry_wrapper(graph_builder.compile(), self.retry_policy)
            return retry_graph

        def llm_invocation(self, state: State) -> State:
            kernel = state["kernel"]
            llm_response = self.llm.invoke(...)
            new_kernel_code = self._extract_and_sanitize_kernel(llm_response)
            ...

        def test_kernel(self, state: State) -> State:
            tests = state["tests"]
            curr_tune_params = state["curr_tune_params"]
            curr_kernel = state["kernel"]
            restrictions = self._ask_restrictions(curr_kernel.code, curr_tune_params)
            tune_result = self._run_tests(curr_kernel, curr_tune_params, tests, restrictions)
            ...
            
You will need to take care of storing the LLM. It will be passed as an argument to ``create_graph``.
The :ref:`Retry Policy <retry_policy>` can be passed to the ``__init__`` of your custom strategy, which should then pass it to ``super().__init__()``. If no policy is provided, a default one will be used.
In the example above the LLM is declared in the ``__init__`` and assigned in ``create_graph``. The retry policy is accepted in ``__init__`` and passed to the base class.

The kernel that needs to be tuned and the tests that need to be executed can be found in the state with the following keys "kernel" and "tests" as shown in the example above.
You will need to execute tests by yourself.
:class:`BaseTuningStrategy <llm_kernel_tuner.tuning_strategies.BaseTuningStrategy>` has multiple helper functions:

#. :func:`_run_tests <llm_kernel_tuner.tuning_strategies.BaseTuningStrategy._run_tests>` tunes the first test provided and compares the output of the other tests. If one of the tests fails an exception will be thrown that could be picked up by a :ref:`Retry Policy<retry_policy>` (available via ``self.retry_policy``). **It is highly advisable to use this function as it implements caching based on the kernel code and tuning parameters, preventing redundant test runs and saving computational resources.**
#. :func:`_extract_and_sanitize_kernel <llm_kernel_tuner.tuning_strategies.BaseTuningStrategy._extract_and_sanitize_kernel>` extracts the kernel from the output of the LLM and removes preprocessor defines.
#. :func:`_ask_restrictions <llm_kernel_tuner.tuning_strategies.BaseTuningStrategy._ask_restrictions>` asks llm restrictions for a given kernel and tuning parameters. Example output: ["block_size_x==block_size_y"] 

You can see them being used in the code above