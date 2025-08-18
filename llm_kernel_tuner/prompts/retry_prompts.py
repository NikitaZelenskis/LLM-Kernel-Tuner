from langchain_core.prompts import PromptTemplate

failed_tests_prompt = "The kernel you provided did not pass the tests. Please try again."

compile_error_prompt = PromptTemplate.from_template("The kernel you provided did not compile. This was the error from the compiler: \"{compiler_error}\" Please try again.")
wrong_tune_params_prompt = "The kernel you provided does not match the original kernel argument list. Please try again, but do not change the arguments"
no_code_prompt = "No code could be extracted from your answer."
timeout_prompt = "The kernel you provided did not finish in a reasonable amount of time. Please try again with a different kernel."
wrong_stucture_prompt = "Your previous answer did not match the expected structure. Please try again but make sure to either call the tool or jsonify the output."
invalid_restrictions_prompt = "One or multiple restrictions that you made are invalid. Most likeley the variable names you used don't match tuning parameters. Try again."

default_error_prompt = PromptTemplate.from_template("There was an error with the kernel you provided. Error: \"{error}\" Please try again.")



invalid_problem_size_prompt = """Problem size could not be extracted. Make sure that the variables you provided exist in the original kernel, they are case sensetive. Try again"""

invalid_output_variables_prompt = """Output variables could not be extracted. Make sure that the variables you provided exist in the original kernel, they are case sensetive. Try again"""

invalid_test_generated_prompt = """The test you provided was invalid. One of the following things was not valid:
1. Input could not be parsed
2. The number of kernel arguments does not match the input array length

Try again"""

invalid_test_generated_with_error_prompt = PromptTemplate.from_template("""
The test you provided was invalid.
Here is the error:
\"\"\"
{error}
\"\"\"
Try again.
""")

default_tester_error_prompt = PromptTemplate.from_template("There was an error when creating the tests. Error: \"{error}\" Please try again.")

default_transformer_error_prompt = "Something went wrong with with your previous answer. Try again"



test_too_long_prompt = PromptTemplate.from_template("""
Input variables for the test you provided are too big and take too long to compute. Make the input smaller.
Current execution time: {current_exec_time}ms
Maximum execution time: {max_exec_time}ms
Min exec time: {min_exec_time}ms
""")

test_too_short_prompt = PromptTemplate.from_template("""
Input variables for the test you provided are too small and cannot be benchmarked on. Make the input bigger.
Current execution time: {current_exec_time}ms
Maximum execution time: {max_exec_time}ms
Min exec time: {min_exec_time}ms
""")

syntax_error_prompt = PromptTemplate.from_template("""
Encountered syntax error while here is the error:
\"\"\"
{syntax_error}
\"\"\"

Please try again with correct syntax
""")

data_size_too_large_prompt = PromptTemplate.from_template("The test you provided is too large to be tested. Maximum size is {max_data_size} bytes, but {data_size} bytes were requested. Please try again with a smaller test.")