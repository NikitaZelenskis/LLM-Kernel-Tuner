from typing import Dict, Any, TypedDict, Optional, List
from llm_kernel_tuner.tunable_kernel import TunableKernel
from llm_kernel_tuner.kernel_test import KernelTest
from llm_kernel_tuner.performance_tracker import PerformanceTracker
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage



class State(TypedDict):
    """A TypedDict representing the state of the kernel tuning process.
    
    This class is used to maintain the state during the kernel tuning process. It contains
    information about the kernel being tuned, current parameters, test cases, performance
    tracking, and other necessary information for the tuning process.
    
    If you are implementing custom tuning/testing strategies, you will need to manage this state
    yourself by implementing proper state transitions and updates in your strategy.
    
    Attributes:
        kernel (TunableKernel): The kernel object that is being tuned, containing the
            kernel code and associated metadata.
        best_params (Optional[Dict[str, Any]]): The best parameters found so far during 
            the tuning process. None if no parameters have been evaluated yet.
        llm (BaseChatModel): The language model chosen by the user for tuning.
        tests (List[KernelTest]): A list of test cases used to validate the kernel 
            during the tuning process.
        messages (List[BaseMessage]): A list of messages exchanged during the tuning
            process, typically for tracking conversation with the LLM.
        curr_tune_params (Dict[str, Any]): The current set of tunable parameters being
            used or considered in the tuning process.
        performance_tracker (PerformanceTracker): Tracks successful optimization steps
            and generates performance overviews for the tuning process."""
    kernel: TunableKernel
    best_params: Optional[Dict[str, Any]]
    llm: Optional[BaseChatModel]
    tests: List[KernelTest]
    messages: List[BaseMessage]
    curr_tune_params: Dict[str, Any]
    performance_tracker: PerformanceTracker
