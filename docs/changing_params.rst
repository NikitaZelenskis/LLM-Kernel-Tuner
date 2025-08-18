Setting tuning strategy parameters
==================================

Autonomous tuning strategy and explicit tuning strategy have some parameters that can be set before starting the kernel.
This page will show some basic examples. Please read documentation for :ref:`Autonomous tuning strategy <autonomous_tuning_strategy>` and :ref:`explicit tuning strategy <explicit_tuning_strategy>` to get a full breakdown of how these strategies work.


Autonomous Strategy examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Disabling breakdown of steps and disable replanning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from llm_kernel_tuner import LLMKernelTransformer
    from llm_kernel_tuner.tuning_strategies import AutonomousTuningStrategy

    kernel_string = "..."

    autonomous_tuning_strategy = AutonomousTuningStrategy(with_replanning=False, breakdown_steps=False)

    kernel_transformer = LLMKernelTransformer(kernel_string, tuning_strategy=autonomous_tuning_strategy)


Changing max breakdown count and max replanning count
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from llm_kernel_tuner import LLMKernelTransformer
    from llm_kernel_tuner.tuning_strategies import AutonomousTuningStrategy

    kernel_string = "..."

    autonomous_tuning_strategy = AutonomousTuningStrategy(max_breakdowns=2, max_replanning=5)

    kernel_transformer = LLMKernelTransformer(kernel_string, tuning_strategy=autonomous_tuning_strategy)




Explicit Strategy example
~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a full example with multiple tuning steps and dependencies:

.. code-block:: python

    from llm_kernel_tuner import LLMKernelTransformer
    from llm_kernel_tuner.tuning_strategies import ExplicitTuningStrategy, TuningStep
    from langchain_core.prompts import PromptTemplate
    from typing import List

    kernel_code = "..."

    my_promtp_template1 = PromptTemplate.from_template("... {kernel_string} ...")
    my_promtp_template2 = PromptTemplate.from_template("... {kernel_string} ...")

    my_tuning_steps: List[TuningStep] = [
        TuningStep(id="tuning_step_id_1", 
            prompt_template=my_promtp_template1,
            tune_params={"block_size_x": [32, 64, 128, 256]}
            skip_evaluation=True
        ),
        TuningStep(id="tuning_step_id_2", 
            prompt_template=my_promtp_template2,
            tune_params={"tune_param": [2, 4, 8]},
            depends_on=["tuning_step_id_1"]
        ),
    ]

    explicit_tuning_strategy = ExplicitTuningStrategy()

    kernel_transformer = LLMKernelTransformer(kernel_code, tuning_strategy=explicit_tuning_strategy)


Visit :ref:`explicit_tuning_strategy` for the breakdown.


