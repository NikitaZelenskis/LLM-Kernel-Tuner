Prompts documentation
=====================

| This page provides an overview of all existing prompts in LLM Kernel Tuner.
| You may overwrite these prompts, but you need to be careful to keep the expected behaviour. This page will explain what each prompt does, where it is used and what it is expected to output.
| If a prompt is of type ``PromptTemplate``, you need to keep it as ``PromptTemplate``. You will also need to include variables in your prompt if it is of type ``PromptTemplate``.
| If a prompt is of type ``str`` it means that the prompt does not have variables and are expected to use string to overwrite the variable.

Lets look at an example of how ``PromptTemplate`` can be overwritten. Lets say we want to overwrite :py:data:`llm_kernel_tuner.prompts.transformer_prompts.kernel_description_prompt` prompt.
:py:data:`llm_kernel_tuner.prompts.transformer_prompts.kernel_description_prompt` is of type ``PromptTemplate`` and has a variable called ``kernel_string``. This means that our new prompt should look something like this:

.. code-block:: python

    from langchain_core.prompts import PromptTemplate

    llm_kernel_tuner.prompts.transformer_prompts.kernel_description_prompt = PromptTemplate.from_template("""
    Give a description for the following CUDA kernel:
    ```CUDA
    {kernel_string}
    ```""")



Transformer prompts
-------------------

| Transformer prompts are prompts used by ``LLMKernelTransformer`` and is located at  ``llm_kernel_tuner.prompts.transformer_prompts``. 
| ``LLMKernelTransformer`` invokes llm only once to get the description of the kernel and therefore only has one prompt.



.. py:data:: llm_kernel_tuner.prompts.transformer_prompts.kernel_description_prompt
    :type: PromptTemplate

    | This prompt is used to get the description of the kernel at the start to help the tuning process.
    | Although not required it is advisable to ask the LLM to not give specific details about code structure as it might interfere with code generation in future prompts.

    **Variables:**

    * kernel_string - kernel that will be tuned


.. py:data:: llm_kernel_tuner.prompts.transformer_prompts.extract_output_var_prompt
    :type: str

    Prompt that is used as system prompt to get the output variables from the kernel.


.. py:data:: llm_kernel_tuner.prompts.transformer_prompts.problem_size_prompt
    :type: str

    Prompt that is used as system prompt to get the ``problem_size`` of a test and kernel.



Tester prompts
--------------

Tester prompts are used by :class:`BaseTestingStrategy <llm_kernel_tuner.testing_strategies.BaseTestingStrategy>` and by :class:`llm_kernel_tuner.testing_strategies.NaiveLLMTester` (see :ref:`testing_strategies`).

.. py:data:: llm_kernel_tuner.prompts.tester_prompts.system_prompt
    :type: str

    | This prompt is used as system prompt for :class:`llm_kernel_tuner.testing_strategies.NaiveLLMTester`.
    | Output of this prompt is expected to generate python code that has the following variable with following types: ``input_data:List[np.ndarray]``. This variable maps to the arguments of the kernel.
    
    For example if this is our kernel that will be tuned:
    
    .. code-block:: cuda

        __global__ void vector_add(float *c, float *a, float *b, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i<n) {
                c[i] = a[i] + b[i];
            }
        }

    This is the expected output of the prompt:

    .. code-block:: python

        n = np.int32(10000000)

        a = np.random.randn(n).astype(np.float32)
        b = np.random.randn(n).astype(np.float32)
        c = np.zeros_like(a)

        input_data = [c, a, b, n]

.. py:data:: llm_kernel_tuner.prompts.tester_prompts.test_prompt
    :type: PromptTemplate
    
    | This prompt is used as user prompt after system prompt for :class:`llm_kernel_tuner.testing_strategies.NaiveLLMTester` to generate test.

    **Variables:**
    
    * kernel_string - Kernel for which to generate test

    See Also:
        :py:data:`llm_kernel_tuner.prompts.tester_prompts.system_prompt`

Tuning strategy prompts
-----------------------

.. py:data:: llm_kernel_tuner.prompts.get_restrictions_prompt
    :type: PromptTemplate

    | Prompt that is used to generate restrictions for a kernel before it is tuned.
    | See `kernel_tuner restrictions <https://kerneltuner.github.io/kernel_tuner/stable/user-api.html#:~:text=restrictions%20(>`_

    **Variables:**
    
    * kernel_string - Kernel for which restrictions will be generated
    * tune_params - Tunable parameters that will be used for the kernel


Autonomous Tuning Strategy prompts
----------------------------------

See :ref:`autonomous_tuning_strategy` for explanation about this strategy.

.. py:data:: llm_kernel_tuner.prompts.autonomous_tuning_strategy.initial_planning_prompt
    :type: PromptTemplate

    | Prompt that will be used to generate the inital planning for the autonomous tuning strategy.
    | This prompt is expected to generate optimization steps.

    **Variables:**
    
    * kernel_string - Kernel that will be tuned.

.. py:data:: llm_kernel_tuner.prompts.autonomous_tuning_strategy.breakdown_step_prompt
    :type: PromptTemplate

    | Prompt that analyzes a CUDA kernel and an optimization step to determine if the step needs to be broken down into smaller steps.
    | This prompt is expected to generate a boolean response (True/False) and optionally a list of smaller steps when applicable.
    | The model will evaluate if the optimization step is at an appropriate granularity level or if it should be divided.

    **Variables:**
    
    * kernel_string - The CUDA kernel code being analyzed.
    * current_step - The optimization step that needs to be evaluated for potential breakdown.

.. py:data:: llm_kernel_tuner.prompts.autonomous_tuning_strategy.agent_prompt
    :type: PromptTemplate

    | Agent prompt that will execute the :ref:`tuning step <tuning_steps>`. 
    | This prompt should only produce the device code.
    | This prompt may introduce tunable parameters, if it does it is expected to give them one of the outputs.

    **Variables:**

    * kernel_string - Kernel that will be tuned.
    * optimization_technique - Current optimization of the tuning step.

.. py:data:: llm_kernel_tuner.prompts.autonomous_tuning_strategy.replan_prompt
    :type: PromptTemplate

    | Prompt used to decide whether a kernel can be further optimized and to generate additional optimization steps.
    | This prompt evaluates the current state of the kernel after previous optimizations and determines if further improvements are possible.
    | It is expected to return a boolean value (True/False) indicating if further optimization is possible, along with a list of additional optimization steps when applicable.

    **Variables:**
    
    * kernel_string - The current state of the CUDA kernel being tuned.
    * past_steps - A list of optimization steps that have already been applied to the kernel.


.. py:data:: llm_kernel_tuner.prompts.autonomous_tuning_strategy.validate_step_prompt
    :type: PromptTemplate

    | Prompt that validates whether an optimization step is applicable to a CUDA kernel.
    | This prompt is expected to evaluate the proposed optimization step against specific criteria and return a boolean result.
    | Expects to return ``True`` or ``False`` of whetehr the optimization step is valid or not.

    By default validates the following criteria of the optimization step:
    
    * Contains only device code modifications.
    * Doesn't require access to host code.
    * Doesn't require compiler flags.
    * Doesn't require external tools or libraries.

    **Variables:**
    
    * optimization_step - The proposed optimization strategy to be validated.

.. py:data:: llm_kernel_tuner.prompts.autonomous_tuning_strategy.fix_params_prompt
    :type: PromptTemplate

    | Prompt used to merge existing and new tunable parameters for a CUDA kernel.
    | This prompt is expected to generate a consolidated dictionary of tunable parameters and potentially adjust the kernel code to use these parameters.

    **Variables:**
    
    * kernel_string - Kernel that will be tuned.
    * existing_tunable_parameters - Dictionary of tunable parameters already defined for the kernel.
    * new_tunable_parameters - Dictionary of new tunable parameters to be merged with existing ones.


Explicit Tuning Strategy prompts
--------------------------------

.. py:data:: llm_kernel_tuner.prompts.explicit_strategy_prompts.system_prompt
    :type: str

    System prompt for :ref:`Explicit Tuning Strategy <explicit_tuning_strategy>`.

.. py:data:: llm_kernel_tuner.prompts.explicit_strategy_prompts.step_evaluation_prompt
    :type: PromptTemplate

    | Prompt that evaluates whether a proposed optimization technique makes sense for a given CUDA kernel.
    | This prompt is expected to generate an assessment of the applicability of the optimization technique.

    **Variables:**
    
    * kernel_string - Kernel that will be tuned.
    * optimization_technique - The proposed optimization technique to be evaluated.


One Prompt Tuning Strategy prompts 
----------------------------------

.. py:data:: llm_kernel_tuner.prompts.one_prompt_strategy_prompts.system_prompt
    :type: str

    System prompt for :ref:`One Prompt tuning strategy <one_prompt_tuning_strategy>`.


.. py:data:: llm_kernel_tuner.prompts.explicit_strategy_prompts.system_prompt
    :type: PromptTemplate

    User prompt that immediately comes after system prompt for :ref:`One Prompt tuning strategy <one_prompt_tuning_strategy>`.

    **Variables:**

    * kernel_string - Kernel that will be tuned.


.. py:data:: llm_kernel_tuner.prompts.one_prompt_strategy_prompts.fix_params_prompt
    :type: PromptTemplate

    | Prompt used to generate values for tunable parameters and incorporate them into the kernel code if needed.
    | This prompt is expected to output a dictionary mapping tunable parameter names to lists of potential values.

    **Variables:**
    
    * kernel_string - The kernel code to be analyzed
    * tunable_parameters - Existing tunable parameters that need to be assigned values


Retry prompts
-------------

These are prompts that are used by default retry policy.

.. note::
    You can also make your own :ref:`retry policy <retry_policy>`.

.. py:data:: llm_kernel_tuner.prompts.retry_prompts.default_error_prompt
    :type: str

    Default error prompt for when LLMKernelTransformer failes when invoking LLM. 

.. py:data:: llm_kernel_tuner.prompts.retry_prompts.failed_tests_prompt
    :type: str

    | Prompt that will be sent to the LLM when the generated kernel fails functional verification tests.


.. py:data:: llm_kernel_tuner.prompts.retry_prompts.compile_error_prompt
    :type: PromptTemplate

    | Prompt that is used when a kernel fails to compile.
    | This prompt is sent to the LLM to request a fixed version of the kernel after compilation errors.

    **Variables:**
    
    * compiler_error - The error message returned by the compiler when attempting to compile the kernel.

.. py:data:: llm_kernel_tuner.prompts.retry_prompts.wrong_tune_params_prompt
    :type: str

    | Prompt that is used when the LLM generates a kernel with modified argument list.


.. py:data:: llm_kernel_tuner.prompts.retry_prompts.no_code_prompt
    :type: str

    | An error message prompt used when the system fails to extract code from the model's response.


.. py:data:: llm_kernel_tuner.prompts.retry_prompts.timeout_prompt
    :type: str
    
    | Prompt that will be sent to the LLM when a kernel execution times out.
    | This prompt is used to inform the LLM that the kernel it provided took too long to execute and needs to be revised.

.. py:data:: llm_kernel_tuner.prompts.retry_prompts.wrong_stucture_prompt
    :type: str

    | Prompt that is used when the LLM generates a kernel with an unexpected structure.
    | Most likeley the LLM did not use tool calling or json output.
    | Depending on LLM model in use this prompt should ask LLM to either use json, structured output or tool calling. 

.. py:data:: llm_kernel_tuner.prompts.retry_prompts.invalid_restrictions_prompt
    :type: str

    | Prompts that is used when llm has failed to generate valid restrictions.
    | Most likely wrong variable names.

.. py:data:: llm_kernel_tuner.prompts.retry_prompts.default_error_prompt
    :type: PromptTemplate

    | Prompt that is used to inform the LLM about errors encountered with a provided kernel.
    | This prompt is expected to relay error information back to the model and request a new attempt.

    **Variables:**
    
    * error - The specific error message that occurred when processing the kernel.


.. py:data:: llm_kernel_tuner.prompts.retry_prompts.invalid_problem_size_prompt
    :type: str

    Prompt that is used when the system fails to extract a valid problem size from the LLM's response.
    This prompt should inform the LLM that the problem size it provided could not be parsed.

.. py:data:: llm_kernel_tuner.prompts.retry_prompts.invalid_output_variables_prompt
    :type: str

    Prompt that is used when the system fails to extract a valid output variables from the LLM's response.
    This prompt should inform the LLM that the output variables it provided could not be parsed.


.. py:data:: llm_kernel_tuner.prompts.retry_prompts.invalid_test_generated_prompt
    :type: str

    Prompt used when the test generated by the LLM is invalid. 
    This could be due to several reasons:

    1. The input  data could not be parsed correctly.
    2. The number of arguments in the kernel does not match the number of provided input arrays.
    
    This prompt should inform the LLM about the invalid test and asks it to provide a valid one.

.. py:data:: llm_kernel_tuner.prompts.retry_prompts.default_tester_error_prompt
    :type: PromptTemplate

    | Prompt that is used when an unspecified error occurs during the test generation process.
    | This prompt relays the specific error message back to the LLM and asks it to try generating the test again.

    **Variables:**

    * error - The specific error message encountered during test generation.
