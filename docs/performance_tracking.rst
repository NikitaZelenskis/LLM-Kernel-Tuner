Performance Tracking
===================

LLM Kernel Tuner provides comprehensive performance tracking capabilities that allow you to monitor and analyze the optimization process. The ``make_kernel_tunable()`` method returns a ``PerformanceTracker`` object that contains detailed information about each successful optimization step.

Return Values
-------------

The ``make_kernel_tunable()`` method returns a tuple with three values:

.. code-block:: python

    tuned_kernel, best_params, performance_tracker = kernel_transformer.make_kernel_tunable()

Where:

- ``tuned_kernel``: The optimized :class:`TunableKernel` object
- ``best_params``: Dictionary containing the best tuning parameters found
- ``performance_tracker``: :class:`PerformanceTracker` object with optimization history

PerformanceTracker Features
---------------------------

The ``PerformanceTracker`` provides several useful methods and properties:

**Key Methods:**

- ``get_total_improvement()``: Returns the total performance improvement percentage from baseline
- ``has_improvements()``: Returns True if any optimization steps were recorded
- ``generate_overview()``: Creates a detailed formatted report of all optimization steps

**Key Properties:**

- ``steps``: List of :class:`PerformanceStep` objects representing each optimization
- ``baseline_time``: The initial execution time before any optimizations

Performance Overview Display
----------------------------

During the tuning process, LLM Kernel Tuner automatically displays a comprehensive performance overview at the end. This overview includes:

- Summary of total optimization steps
- Baseline vs. final execution times
- Total improvement percentage and speedup factor
- Detailed breakdown of each optimization step
- Tunable parameters and restrictions for each step
- Best parameter values found

Example Usage
-------------

Here's a complete example showing how to use the performance tracking features:

.. code-block:: python

    from llm_kernel_tuner import LLMKernelTransformer
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model_name='gpt-4o-mini')
    
    kernel_string = '''
    __global__ void vectorAdd(float *A, float *B, float *C, int N) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < N) {
            C[idx] = A[idx] + B[idx];
        }
    }
    '''

    kernel_transformer = LLMKernelTransformer(kernel_string, model)
    
    # Get all three return values
    tuned_kernel, best_params, performance_tracker = kernel_transformer.make_kernel_tunable()
    
    # Access performance information
    print(f"Optimization steps completed: {len(performance_tracker.steps)}")
    
    if performance_tracker.has_improvements():
        total_improvement = performance_tracker.get_total_improvement()
        print(f"Total performance improvement: {total_improvement:.2f}%")
        
        # Access individual optimization steps
        for i, step in enumerate(performance_tracker.steps, 1):
            print(f"Step {i}: {step.step_description}")
            print(f"  Improvement: {step.improvement_percentage:.2f}%")
            print(f"  Execution time: {step.new_execution_time:.6f}s")
    
    # Generate detailed overview (already displayed during tuning)
    overview = performance_tracker.generate_overview()
    # print(overview)  # Uncomment to display again

PerformanceStep Details
-----------------------

Each optimization step is represented by a :class:`PerformanceStep` object containing:

- ``step_description``: Human-readable description of the optimization
- ``kernel_code``: The optimized kernel code after this step
- ``old_execution_time``: Previous best execution time (None for first step)
- ``new_execution_time``: New execution time after optimization
- ``improvement_percentage``: Calculated improvement percentage
- ``tunable_parameters``: The tunable parameters used for this step
- ``restrictions``: Parameter restrictions applied during tuning
- ``best_tune_params``: The best parameter values found for this kernel
- ``timestamp``: When this step was recorded

Integration with Existing Code
------------------------------

If you have existing code that uses the old two-value return format, you can easily update it:

**Old format:**

.. code-block:: python

    tuned_kernel, best_params = kernel_transformer.make_kernel_tunable()

**New format:**

.. code-block:: python

    tuned_kernel, best_params, performance_tracker = kernel_transformer.make_kernel_tunable()
    
    # The performance_tracker is now available for additional analysis
    # The performance overview is automatically displayed during tuning

This change is backward-compatible in the sense that the first two return values remain the same, but you'll need to update your code to handle the third return value.