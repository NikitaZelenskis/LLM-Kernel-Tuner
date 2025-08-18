import re
from langchain_core.outputs import ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class ThinkingStripperWrapper(BaseChatOpenAI):
    """
    A wrapper for a BaseChatModel that strips <think>...</think> tags
    from the LLM's output before returning it.

    Because this class inherits from BaseChatModel, it can be used as a
    drop-in replacement wherever a BaseChatModel is expected.
    """
    llm: BaseChatModel
    pattern: str

    def __init__(self, **data: Any):
        """
        Initializes the wrapper and then overrides the model_name attribute
        on the instance with a sanitized version.
        """
        if 'api_key' not in data:
            data['api_key'] = 'dummy-key'
        super().__init__(**data)
        
        if hasattr(self.llm, 'model_name'):
            self.model_name = self.llm.model_name

    def _strip_thinking(self, text: str) -> str:
        """Removes <think>...</think> (or other pattern) blocks and surrounding whitespace."""
        return re.sub(self.pattern, "", text, flags=re.DOTALL).strip()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Calls the wrapped LLM's _generate method and cleans the output.
        """
        # Get the original result from the wrapped LLM
        result = self.llm._generate(messages, stop, run_manager, **kwargs)

        # Clean the content of each generation in the result
        for generation in result.generations:
            if hasattr(generation, 'message') and isinstance(generation.message.content, str):
                cleaned_content = self._strip_thinking(generation.message.content)
                generation.message.content = cleaned_content
        
        return result

    
    def __getattr__(self, name: str) -> Any:
        """
        Fallback for any other attribute that's not explicitly defined on the
        wrapper. This makes accessing things like `temperature`, `base_url`, etc.,
        work transparently.
        """
        # This is safe from recursion because `self.llm` and `self.pattern`
        # are found on the instance before __getattr__ is ever called.
        return getattr(self.llm, name)

    # --- Required property for LangChain ---
    @property
    def _llm_type(self) -> str:
        """Return the type of the LLM."""
        return "thinking-stripper-wrapper"