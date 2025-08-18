from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate

class TuningStep:
    """Initialize a TuningStep instance.
    
    A TuningStep represents a single optimization technique to be applied to a kernel.
    Each step has a unique identifier, a prompt template to guide the LLM in applying
    the optimization, optional parameters to tune, and dependency information.
    
    Args:
        id (str): A unique identifier for this tuning step.
        prompt_template (PromptTemplate): The prompt template containing instructions for the LLM
            on how to implement this optimization technique.
        tune_params (Dict[str, List[Any]], optional): A dictionary mapping parameter names to lists of possible values
            to explore during tuning. Each parameter will be grid-searched across
            its possible values. Defaults to an empty dictionary.
        depends_on (List[str], optional): A list of tuning step IDs that must be completed before this
            step can be executed. Allows for defining dependencies between
            optimization techniques. Defaults to an empty list.
        skip_evaluation (bool, optional): If True, the system will not ask the LLM to evaluate whether
            this step is necessary and will always execute it. Defaults to False.
    """
    def __init__(self, id:str, 
                 prompt_template: PromptTemplate, 
                 tune_params: Dict[str, List[Any]] = {},
                 depends_on: List[str] = [],
                 skip_evaluation: bool = False):
        self.id = id
        self.prompt_template = prompt_template
        self.tune_params = tune_params
        self.depends_on = depends_on
        # If true, the tuning step will skip asking llm to evaluate whether the stpe needs to run and will always execute this step
        self.skip_evaluation = skip_evaluation