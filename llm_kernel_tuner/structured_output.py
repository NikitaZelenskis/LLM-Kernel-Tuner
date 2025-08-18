from llm_kernel_tuner.llm_kernel_tuner_logger import get_logger
from llm_kernel_tuner.retry import RetryPolicy, create_retry_wrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from typing import Type, List, Union, Dict, Any, TypedDict, Optional
from pydantic import BaseModel
from enum import Enum
import re

class StructuredOutputType(Enum):
    """
    Enum defining the types of structured output methods available for LLMs.
    
    This enum is used to configure how structured output should be handled
    for different language models, particularly in the LLM metadata to indicate
    which approach should be used.
    
    Attributes:
        TOOL_CALLING: Use the native tool calling mechanism for structured output.
                     This is the preferred method for LLMs that support it natively
                     (e.g., OpenAI GPT models, Anthropic Claude). It leverages the
                     model's built-in function calling capabilities to ensure
                     properly structured responses.
                     
        SEPARATE_REQUEST: Use a separate request with prompt engineering to achieve
                         structured output. This method uses the StructuredOutputEmulator
                         to parse and structure the LLM's text response into the desired
                         format. This is used as a fallback for models that don't
                         support native structured output or tool calling.
        HYBRID_JSONIFY: A two-step process where the first LLM call generates raw text,
                       and a second call uses the model's native structured output
                       capability to format the text into JSON. LLMs that don't allow thinking .

        JSON_SCHEMA: Use the model's native JSON schema if available. This leverages
                  built-in JSON formatting capabilities that some models provide
                  (e.g., OpenAI's response_format="json_schema"). It ensures the
                  model outputs valid JSON without requiring tool calling or
                  extensive prompt engineering.
                  
        DEFAULT: Use the model's default structured output behavior. This allows
                the model to automatically determine the best approach for structured
                output based on its capabilities. The model will use its native
                .with_structured_output() method without specifying a particular
                method, letting the implementation choose the optimal strategy.
    
    Usage:
        This enum is typically used as a parameter in the LLMKernelTransformer constructor:
        
        .. code-block:: python
        
            from llm_kernel_tuner import LLMKernelTransformer
            from llm_kernel_tuner.structured_output import StructuredOutputType
            
            # For models with tool calling support
            kernel_transformer = LLMKernelTransformer(
                kernel_string,
                llm,
                structured_output_type=StructuredOutputType.TOOL_CALLING
            )
            
            # For models with JSON mode support
            kernel_transformer = LLMKernelTransformer(
                kernel_string,
                llm,
                structured_output_type=StructuredOutputType.JSON_SCHEMA
            )
            
            # For models using default behavior
            kernel_transformer = LLMKernelTransformer(
                kernel_string,
                llm,
                structured_output_type=StructuredOutputType.DEFAULT
            )
            
            # For models without native structured output
            kernel_transformer = LLMKernelTransformer(
                kernel_string,
                llm,
                structured_output_type=StructuredOutputType.SEPARATE_REQUEST
            )

                        
            # For models that benefit from two-step generation and formatting
            kernel_transformer = LLMKernelTransformer(
                kernel_string,
                llm,
                structured_output_type=StructuredOutputType.HYBRID_JSONIFY
            )
    
    Note:
        The LLMKernelTransformer automatically configures the LLM's metadata based on
        this parameter. Users should not manually set the structured_output_type in
        the LLM's metadata.
    """
    TOOL_CALLING = "tool_calling"
    SEPARATE_REQUEST = "separate_request"
    JSON_SCHEMA = "json_schema"
    HYBRID_JSONIFY = "hybrid_jsonify"
    DEFAULT = "default"

logger = get_logger(__name__)


JSONIFY_PROMPT_TEMPLATE = PromptTemplate.from_template("""
You are an expert at parsing and structuring text. Your task is to take the user-provided text and convert it into a JSON object that strictly adheres to the provided JSON schema.

Do not add any fields that are not in the schema.
Do not omit any required fields.
Your output must be ONLY the JSON object surrounded by (```json ... ```) fence, with no other text or commentary.
The JSON object must be syntactically correct. Pay close attention to commas, quotes, and brackets.


JSON Schema:
```json
{format_instructions}
```

Text to convert:
\"\"\"
{raw_text}
\"\"\"
""")


default_retry_prompt = """
Could not parse your previous response, please try again.
Make sure your answer is a correct json object with no other text or commentary.
"""




def _default_handler(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
    state["messages"].append(HumanMessage(default_retry_prompt))
    return state

retry_policy = RetryPolicy(
    max_retries=2,
    default_handler=_default_handler,
)

class RetryState(TypedDict):
    messages: List[BaseMessage]
    text_to_convert: str
    parsed_object: Optional[BaseModel]
    kwargs: Dict[str, Any]

class StructuredOutputEmulator:
    """
    A wrapper to emulate the .with_structured_output() method for LLMs
    that do not support it natively by using a prompt and a parser.
    """
    def __init__(self, llm: BaseChatModel, pydantic_object: Type[BaseModel], use_native_jsonify: bool = False):
        self.llm = llm
        self.pydantic_object = pydantic_object
        self.parser = PydanticOutputParser(pydantic_object=self.pydantic_object)
        self.use_native_jsonify = use_native_jsonify
        
        self.str_output_chain = self.llm | StrOutputParser()

        if self.use_native_jsonify:
            self.json_chain = self.llm.with_structured_output(self.pydantic_object)
        else:
            self.json_chain = self.str_output_chain

        self.format_instructions = self.parser.get_format_instructions(),

    
    def invoke(self, messages: Union[str, List[BaseMessage]], **kwargs):
        """
        Executes the two-call chain.
        """
        if isinstance(messages, str):
            logger.debug("Input is a string. Wrapping in a HumanMessage for processing.")
            messages = [HumanMessage(content=messages)]
        elif not isinstance(messages, list):
             raise TypeError("Input must be a string or a list of BaseMessages.")
        # --- CALL 1: Generate unstructured content ---
        logger.debug("Starting first call (generation)...")
        unstructured_text = self.str_output_chain.invoke(messages, **kwargs)
        # --- CALL 2: Format the content into JSON ---
        logger.debug("Starting second call (formatting)...")
        retry_graoh = create_retry_wrapper(self.jsonify, retry_policy, RetryState)
        
        init_state: RetryState = {
            "messages": [],
            "text_to_convert": unstructured_text,
            "parsed_object" : None,
            "kwargs": kwargs,
        }
        new_state: RetryState = retry_graoh.invoke(init_state) #type:ignore
        if new_state["parsed_object"] is None:
            raise ValueError("Failed to jsonify LLM response")

        return new_state["parsed_object"]


    def jsonify(self, state: RetryState) -> RetryState:   
        jsonify_prompt = JSONIFY_PROMPT_TEMPLATE.format(format_instructions=self.format_instructions, raw_text=state["text_to_convert"])
        
        if self.use_native_jsonify:
            try:
                logger.debug("Attempting to jsonify using native structured output.")
                parsed_object = self.json_chain.invoke(jsonify_prompt, **state["kwargs"])
                state["parsed_object"] = parsed_object # type: ignore
                return state
            except Exception as e:
                # Fallback to string-based parsing on failure
                logger.warning(f"Native JSON-ification failed: {e}. Falling back to string parsing.")

        if len(state["messages"]) == 0:
            state['messages'] = [
                HumanMessage(jsonify_prompt)
            ]

        raw_response = self.str_output_chain.invoke(state["messages"], **state["kwargs"])
        state['messages'].append(AIMessage(raw_response))
        try:
            pattern = r"```json(.*?)```"
            matches = re.findall(pattern, raw_response, re.DOTALL)

            if not matches:
                logger.debug("No JSON fences found in the output. Attempting to parse the entire response.")
                json_string = raw_response
            else:
                json_string = matches[-1].strip()
                logger.debug(f"Found {len(matches)} JSON blocks. Extracting the last one.")
            
            state["parsed_object"] = self.parser.parse(json_string)
            return state

        except Exception as e:
            raise OutputParserException(
                f"Failed to parse JSON. Raw formatted response:\n---\n{raw_response}\n---",
                llm_output=raw_response
            ) from e


def get_structured_llm(
    llm: BaseChatModel,
    pydantic_schema: Type[BaseModel]
):
    """
    Factory function that returns a structured output-capable runnable.

    It inspects the LLM's metadata to decide whether to use the native
    `.with_structured_output()` method or fall back to the emulation wrapper.
    The metadata is set by :class:`LLMKernelTransformer <llm_kernel_tuner.LLMKernelTransformer>` constructor.

    Args:
        llm: The language model instance. It should have a `metadata` dict.
        pydantic_schema: The Pydantic model for the desired output.

    Returns:
        A LangChain runnable that will produce a structured Pydantic object.
    """
    structured_output_type = (
        llm.metadata.get("structured_output_type", StructuredOutputType.DEFAULT)
        if llm.metadata else StructuredOutputType.DEFAULT
    )


    if structured_output_type == StructuredOutputType.TOOL_CALLING:
        return llm.with_structured_output(pydantic_schema, method="function_calling")
    elif structured_output_type == StructuredOutputType.JSON_SCHEMA:
        return llm.with_structured_output(pydantic_schema, method="json_schema")
    elif structured_output_type == StructuredOutputType.HYBRID_JSONIFY:
        return StructuredOutputEmulator(llm, pydantic_schema, use_native_jsonify=True)
    elif structured_output_type == StructuredOutputType.DEFAULT:
        return llm.with_structured_output(pydantic_schema)
    else:
        return StructuredOutputEmulator(llm, pydantic_schema)