API documentation
=================

.. autoclass:: llm_kernel_tuner.LLMKernelTransformer
   :members: make_kernel_tunable, add_test

.. autoclass:: llm_kernel_tuner.TunableKernel
   :members: copy, tune, test, get_arg_position

.. autoclass:: llm_kernel_tuner.tuning_strategies.BaseTuningStrategy
   :members: create_graph, _run_tests, _extract_and_sanitize_kernel, _ask_restrictions


.. autoclass:: llm_kernel_tuner.tuning_strategies.AutonomousTuningStrategy

.. autoclass:: llm_kernel_tuner.tuning_strategies.ExplicitTuningStrategy

.. autoclass:: llm_kernel_tuner.tuning_strategies.TuningStep

.. autoclass:: llm_kernel_tuner.KernelTest

.. autoclass:: llm_kernel_tuner.testing_strategies.BaseTestingStrategy
   :members: get_test_from_code

.. autoclass:: llm_kernel_tuner.tuning_state.State

.. autoclass:: llm_kernel_tuner.PerformanceTracker
   :members: record_step, set_baseline_time, get_total_improvement, generate_overview, has_improvements

.. autoclass:: llm_kernel_tuner.PerformanceStep
   
.. autoclass:: llm_kernel_tuner.retry.RetryPolicy

.. autofunction:: llm_kernel_tuner.retry.create_retry_wrapper
