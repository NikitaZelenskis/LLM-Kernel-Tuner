#!/usr/bin/env python3

import argparse
import json
import sys
import re
import ast

def extract_code(text: str) -> str:
    """Extracts code from a string, typically from markdown-style backticks."""
    # This regex handles both ```...``` and ```cuda ...``` formats
    match = re.search(r'```(?:\w*\n)?(.*)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Return original text if no code block found

class TuningReportGenerator:
    """Parses a JSONL log file and generates a tuning report."""

    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.log_entries = []
        self.summary = {
            "initial_code": None,
            "initial_time": None,
            "initial_params": None,
            "plan": [],
            "steps": [],
            "breakdowns": [],          # <-- NEW: To store breakdown events
            "replans": [],             # <-- NEW: To store replan events
            "final_kernel": None,
            "final_params": None,
            "final_time": None
        }
        self.state = {
            "best_time": float('inf'),
            "best_code": None,
            "current_step": None,
            "current_step_code_try": None,
            "pending_breakdown": None, # <-- NEW: State for multi-line breakdown parsing
        }

    def parse_log(self):
        """Reads and processes the entire log file."""
        try:
            with open(self.log_file_path, 'r') as f:
                for line in f:
                    try:
                        self.log_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            print(f"Error: Log file not found at '{self.log_file_path}'", file=sys.stderr)
            sys.exit(1)

        for entry in self.log_entries:
            if "Invoking llm to get kernel description for:" in entry.get("message", ""):
                self.summary["initial_code"] = extract_code(entry.get("message", ""))
                self.state["best_code"] = self.summary["initial_code"]
                break
        
        for entry in self.log_entries:
            self._process_entry(entry)
        
        self._finalize_step_if_pending()

    def _process_entry(self, entry: dict):
        """Processes a single log entry and updates the state."""
        msg = entry.get("message", "")
        timestamp = entry.get("timestamp") # <-- MODIFIED: Capture timestamp for every entry

        if "Initial kernel time:" in msg:
            time_match = re.search(r'Initial kernel time: ([\d.eE+-]+)', msg)
            params_match = re.search(r'tuning parameters: (.*)', msg)
            if time_match:
                initial_time = float(time_match.group(1))
                self.summary["initial_time"] = initial_time
                self.state["best_time"] = initial_time
            if params_match:
                self.summary["initial_params"] = params_match.group(1)

        elif "LLM answered with plan:" in msg:
            plan_str_match = re.search(r'LLM answered with plan: "(.*)"', msg)
            if plan_str_match:
                plan_string = plan_str_match.group(1)
                try:
                    self.summary["plan"] = ast.literal_eval(plan_string)
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse plan string: {plan_string} ({e})", file=sys.stderr)

        # <-- NEW: Handle breakdown events -->
        elif "Asking for breakdown of step:" in msg:
            match = re.search(r'Asking for breakdown of step: "(.*)"', msg)
            if match:
                self.state["pending_breakdown"] = {
                    "type": "breakdown",
                    "timestamp": timestamp,
                    "description": match.group(1),
                }
        elif "LLM answered with breakdown:" in msg and self.state["pending_breakdown"]:
            match = re.search(r"breakdown:\s*(True|False),\s*steps:\s*(\[.*\])", msg, re.DOTALL)
            if match:
                breakdown_succeeded = match.group(1) == "True"
                new_steps_str = match.group(2)
                self.state["pending_breakdown"]["succeeded"] = breakdown_succeeded
                if breakdown_succeeded:
                    try:
                        self.state["pending_breakdown"]["new_steps"] = ast.literal_eval(new_steps_str)
                    except (ValueError, SyntaxError):
                        self.state["pending_breakdown"]["new_steps"] = ["Error parsing steps"]
                self.summary["breakdowns"].append(self.state["pending_breakdown"])
                self.state["pending_breakdown"] = None
        
        # <-- NEW: Handle replan events -->
        elif "Asking llm for replan" in msg:
            # We don't create a pending state as the answer is expected in the next log
            pass
        elif "LLM response: replan:" in msg:
            match = re.search(r"replan:\s*(True|False),\s*steps:\s*(\[.*\])", msg, re.DOTALL)
            if match:
                replan_event = {
                    "type": "replan",
                    "timestamp": timestamp,
                    "succeeded": match.group(1) == "True"
                }
                if replan_event["succeeded"]:
                    try:
                        replan_event["new_plan"] = ast.literal_eval(match.group(2))
                    except (ValueError, SyntaxError):
                        replan_event["new_plan"] = ["Error parsing new plan"]
                self.summary["replans"].append(replan_event)

        elif "Applying optimization step:" in msg:
            self._finalize_step_if_pending()
            step_description = re.search(r'Applying optimization step: "(.*)"', msg).group(1)
            self.state["current_step"] = {
                "type": "step", # <-- NEW: Add type for sorting
                "timestamp": timestamp, # <-- NEW: Add timestamp for sorting
                "description": step_description,
                "outcome": "Unknown",
                "code_before": self.state["best_code"],
                "code_after": None,
                "error": None,
                "tries": 0
            }

        elif "LLM answered with tuned kernel:" in msg and self.state["current_step"]:
            code_match = re.search(r'LLM answered with tuned kernel: "(.*)" and tunable parameters:', msg, re.DOTALL)
            if code_match:
                raw_code_string = code_match.group(1)
                self.state["current_step_code_try"] = extract_code(raw_code_string)
                self.state["current_step"]["tries"] += 1

        elif "function/subgraph failed with the following error:" in msg and self.state["current_step"]:
            error_match = re.search(r'\(text\): (.*)', msg)
            if error_match:
                self.state["current_step"]["outcome"] = "Failure"
                self.state["current_step"]["error"] = error_match.group(1)
                self.state["current_step"]["code_after"] = self.state["current_step_code_try"]

        elif "New best kernel has been chosen," in msg:
            # Handle case where a new best is found inside or outside a step
            new_time_str = re.search(r'execution time: ([\d.eE+-]+)', msg).group(1)
            new_time = float(new_time_str)
            new_code_match = re.search(r'kernel code: ```(.*)``` with execution time:', msg, re.DOTALL)
            if not new_code_match:
                 new_code_match = re.search(r'kernel code: (.*) with execution time:', msg, re.DOTALL)
            
            new_code = extract_code(new_code_match.group(1))
            
            if self.state["current_step"]:
                self.state["current_step"]["outcome"] = "Success"
                self.state["current_step"]["new_time"] = new_time
                self.state["current_step"]["code_after"] = new_code
                self.state["current_step"]["previous_time"] = self.state["best_time"]
                
                self.state["best_time"] = new_time
                self.state["best_code"] = new_code
                self._finalize_step_if_pending()
            
        elif "Final best performaing kernel:" in msg:
            self._finalize_step_if_pending()
            final_kernel_match = re.search(r'Final best performaing kernel: `\n(.*?)\n` with following', msg, re.DOTALL)
            if final_kernel_match:
                kernel_str = final_kernel_match.group(1)
                code_match = re.search(r'code=(.*?), args=', kernel_str, re.DOTALL)
                time_match = re.search(r'best_time=([\d.eE+-]+)', kernel_str)

                if code_match: self.summary["final_kernel"] = code_match.group(1).strip().replace('\\n', '\n')
                if time_match: self.summary["final_time"] = float(time_match.group(1))

            params_match = re.search(r'tunable parameters: `\n(.*?)\n`', msg, re.DOTALL)
            if params_match:
                self.summary["final_params"] = params_match.group(1).strip()

    def _finalize_step_if_pending(self):
        """Checks if a step is in progress and finalizes it."""
        if not self.state["current_step"]:
            return
        
        step = self.state["current_step"]
        if step["outcome"] == "Unknown":
            step["outcome"] = "No Improvement"
            step["code_after"] = self.state["current_step_code_try"]
        
        self.summary["steps"].append(step)
        self.state["current_step"] = None
        self.state["current_step_code_try"] = None

    def print_report(self):
        """Prints the final formatted report to the console."""
        print("="*80)
        print(" CUDA Kernel Tuning Report")
        print("="*80)

        if self.summary["initial_code"]:
            print("\n--- Initial Kernel ---")
            print(self.summary["initial_code"])
            if self.summary["initial_time"] is not None:
                initial_time_ms = self.summary["initial_time"] * 1000
                print(f"\nInitial Execution Time: {initial_time_ms:.4f} ms")
                if self.summary["initial_params"]:
                    print(f"Initial Parameters: {self.summary['initial_params']}")
        else:
            print("\nCould not determine initial kernel code from log.", file=sys.stderr)
            return

        if self.summary["plan"]:
            print("\n--- Initial LLM Optimization Plan ---")
            for i, step_plan in enumerate(self.summary["plan"], 1):
                print(f"{i}. {step_plan}")
        
        print("\n" + "="*80)
        print(" Chronological Tuning Log")
        print("="*80)
        
        # <-- MODIFIED: Create a single, sorted list of all events -->
        all_events = self.summary['steps'] + self.summary['breakdowns'] + self.summary['replans']
        all_events.sort(key=lambda x: x.get('timestamp', ''))

        step_counter = 0
        for event in all_events:
            if event['type'] == 'step':
                step_counter += 1
                self._print_step_details(event, step_counter)
            elif event['type'] == 'breakdown':
                self._print_breakdown_details(event)
            elif event['type'] == 'replan':
                self._print_replan_details(event)


        print("\n" + "="*80)
        print(" Final Result")
        print("="*80)

        final_best_time = self.summary.get('final_time') or (self.state['best_time'] if self.state['best_time'] != float('inf') else None)
        
        if self.summary["final_kernel"] and final_best_time is not None:
            final_time_ms = final_best_time * 1000
            print("\n--- Best Performing Kernel ---")
            print(f"Final Execution Time: {final_time_ms:.4f} ms")
            print(f"Best Tunable Parameters: {self.summary['final_params']}")
            print("\n```cuda")
            print(self.summary["final_kernel"])
            print("```")
        elif self.state['best_code'] and final_best_time is not None:
            final_time_ms = final_best_time * 1000
            print("\n--- Best Performing Kernel (from last successful step) ---")
            print(f"Final Execution Time: {final_time_ms:.4f} ms")
            print(f"Best Tunable Parameters: {self.summary.get('final_params', 'Not explicitly logged at end')}")
            print("\n```cuda")
            print(self.state['best_code'])
            print("```")
        else:
            print("\nCould not determine the final best kernel from the logs.")

    # <-- NEW: Helper functions to print different event types -->
    def _print_step_details(self, step: dict, counter: int):
        header = f"--- Step {counter}: {step['description']} ---"
        print(f"\n{header}\n")
        print(f"Outcome: {step['outcome']}")

        if step["outcome"] == "Success":
            prev_time = step["previous_time"]
            new_time = step["new_time"]
            print(f"  - Previous Best Time: {'N/A' if prev_time == float('inf') else f'{prev_time * 1000:.4f} ms'}")
            print(f"  - New Best Time:      {new_time * 1000:.4f} ms")
            if prev_time != float('inf') and prev_time > 0:
                improvement = ((prev_time - new_time) / new_time) -1 if new_time == 0 else ((prev_time - new_time) / prev_time) * 100
                print(f"  - Improvement:        {improvement:.2f}%")
            print("\n  Resulting Code:")
            print(f"```cuda\n{step['code_after']}\n```")

        elif step["outcome"] == "Failure":
            print(f"  - Error after {step['tries']} trie(s): {step['error']}")
            if step.get('code_after'):
                    print(f"\n  Final Code Attempt that Failed:")
                    print(f"```cuda\n{step['code_after']}\n```")

        elif step["outcome"] == "No Improvement":
            print("  - The changes did not result in a faster kernel.")
            if step.get("code_after"):
                print(f"\n  Tested Code:")
                print(f"```cuda\n{step['code_after']}\n```")
        
        print("-" * len(header))

    def _print_breakdown_details(self, breakdown: dict):
        header = f">>> Breakdown Attempt on Step: \"{breakdown['description']}\" <<<"
        print(f"\n{header}\n")
        if breakdown['succeeded']:
            print("Outcome: SUCCESS")
            print("  - The step was broken down into smaller, more manageable steps.")
            print("  - New sub-steps to be attempted:")
            for i, sub_step in enumerate(breakdown.get('new_steps', []), 1):
                print(f"    {i}. {sub_step}")
        else:
            print("Outcome: NO BREAKDOWN")
            print("  - The LLM decided not to break this step down further.")
        print("-" * len(header))

    def _print_replan_details(self, replan: dict):
        header = ">>> Replanning Attempt <<<"
        print(f"\n{header}\n")
        if replan['succeeded']:
            print("Outcome: SUCCESS")
            print("  - A new optimization plan has been generated.")
            print("  - New Plan:")
            for i, step in enumerate(replan.get('new_plan', []), 1):
                print(f"    {i}. {step}")
        else:
            print("Outcome: NO REPLAN")
            print("  - The LLM decided to continue with the existing plan.")
        print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(
        description="Parse a JSONL log file from a kernel tuning process and generate a summary report.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("logfile", help="Path to the .jsonl log file.")
    args = parser.parse_args()

    report_generator = TuningReportGenerator(args.logfile)
    report_generator.parse_log()
    report_generator.print_report()

if __name__ == "__main__":
    main()