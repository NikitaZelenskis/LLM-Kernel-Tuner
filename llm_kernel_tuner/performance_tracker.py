"""Performance tracking components for LLM Kernel Tuner.

This module provides data structures and functionality to track and display
successful optimization steps during the kernel tuning process.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class PerformanceStep:
    """Data class representing a successful optimization step.
    
    This class captures comprehensive information about each accepted optimization
    step, including the step description, kernel code changes, performance metrics,
    tunable parameters, and restrictions.
    
    Attributes:
        step_description: Human-readable description of the optimization step
        kernel_code: The optimized kernel code after this step
        old_execution_time: Previous best execution time (None for first step)
        new_execution_time: New execution time after optimization
        improvement_percentage: Calculated improvement percentage
        tunable_parameters: The tunable parameters used for this step
        restrictions: Parameter restrictions applied during tuning
        best_tune_params: The best parameter values found for this kernel
        timestamp: When this step was recorded
    """
    step_description: str
    kernel_code: str
    old_execution_time: Optional[float]
    new_execution_time: float
    improvement_percentage: float
    tunable_parameters: Dict[str, List[Any]]
    restrictions: List[str]
    best_tune_params: Dict[str, Any]
    timestamp: datetime


class PerformanceTracker:
    """Tracks and manages successful optimization steps during kernel tuning.
    
    This class provides functionality to record successful optimization steps,
    calculate performance improvements, and generate formatted overviews of
    the tuning process results.
    """
    
    def __init__(self):
        """Initialize a new PerformanceTracker instance."""
        self.steps: List[PerformanceStep] = []
        self.baseline_time: Optional[float] = None
    
    def record_step(self, step: PerformanceStep) -> None:
        """Record a successful optimization step.
        
        Args:
            step: The PerformanceStep to record
        """
        self.steps.append(step)
    
    def set_baseline_time(self, time: float) -> None:
        """Set the baseline execution time before any optimizations.
        
        Args:
            time: The initial kernel execution time
        """
        self.baseline_time = time
    
    def get_total_improvement(self) -> float:
        """Calculate the total performance improvement from baseline.
        
        Returns:
            The total improvement percentage from baseline to final result.
            Returns 0.0 if no baseline is set or no steps are recorded.
        """
        if not self.baseline_time or not self.steps:
            return 0.0
        
        final_time = self.steps[-1].new_execution_time
        return ((self.baseline_time - final_time) / self.baseline_time) * 100.0
    
    def generate_overview(self) -> str:
        """Generate a formatted overview of all recorded optimization steps.
        
        Returns:
            A formatted string containing the performance overview, or a message
            indicating no improvements were found if no steps were recorded.
        """
        if not self.steps:
            return "No performance improvements were found during the tuning process."
        
        overview_lines = []
        overview_lines.append("=" * 80)
        overview_lines.append("PERFORMANCE OPTIMIZATION OVERVIEW")
        overview_lines.append("=" * 80)
        overview_lines.append("")
        
        # Summary section with comprehensive metrics
        overview_lines.append("SUMMARY")
        overview_lines.append("-" * 40)
        overview_lines.append(f"Total optimization steps: {len(self.steps)}")
        
        if self.baseline_time:
            final_time = self.steps[-1].new_execution_time
            total_improvement = self.get_total_improvement()
            speedup_factor = self.baseline_time / final_time if final_time > 0 else float('inf')
            
            overview_lines.append(f"Baseline execution time: {self.baseline_time:.6f} seconds")
            overview_lines.append(f"Final execution time:    {final_time:.6f} seconds")
            overview_lines.append(f"Total improvement:       {total_improvement:.2f}%")
            overview_lines.append(f"Speedup factor:          {speedup_factor:.2f}x")
        else:
            final_time = self.steps[-1].new_execution_time
            overview_lines.append(f"Final execution time:    {final_time:.6f} seconds")
            overview_lines.append("(No baseline time available for total improvement calculation)")
        
        overview_lines.append("")
        
        # Detailed step-by-step breakdown
        overview_lines.append("OPTIMIZATION STEPS")
        overview_lines.append("-" * 40)
        overview_lines.append("")
        
        cumulative_improvement = 0.0
        for i, step in enumerate(self.steps, 1):
            # Step header with enhanced formatting
            overview_lines.append(f"┌─ Step {i}: {step.step_description}")
            overview_lines.append("│")
            
            # Performance metrics section
            overview_lines.append("│  ⏱️  PERFORMANCE METRICS")
            if step.old_execution_time:
                overview_lines.append(f"│     Previous time:     {step.old_execution_time:.6f} seconds")
                overview_lines.append(f"│     New time:          {step.new_execution_time:.6f} seconds")
                time_saved = step.old_execution_time - step.new_execution_time
                overview_lines.append(f"│     Time saved:        {time_saved:.6f} seconds")
            else:
                overview_lines.append(f"│     Execution time:    {step.new_execution_time:.6f} seconds")
                overview_lines.append("│     (Initial optimization - no previous time)")
            
            overview_lines.append(f"│     Step improvement:  {step.improvement_percentage:.2f}%")
            
            # Calculate cumulative improvement from baseline
            if self.baseline_time and step.new_execution_time > 0:
                cumulative_improvement = ((self.baseline_time - step.new_execution_time) / self.baseline_time) * 100.0
                overview_lines.append(f"│     Cumulative gain:   {cumulative_improvement:.2f}%")
            
            overview_lines.append("│")
            
            # Tunable parameters section
            if step.tunable_parameters:
                overview_lines.append("│  🔧 TUNABLE PARAMETERS")
                for param, values in step.tunable_parameters.items():
                    # Format parameter values nicely
                    if isinstance(values, list) and len(values) <= 10:
                        values_str = str(values)
                    elif isinstance(values, list) and len(values) > 10:
                        values_str = f"[{values[0]}, {values[1]}, ..., {values[-1]}] ({len(values)} values)"
                    else:
                        values_str = str(values)
                    overview_lines.append(f"│     {param:<20} {values_str}")
                overview_lines.append("│")
            
            # Best parameter values section
            if step.best_tune_params:
                overview_lines.append("│  ⭐ OPTIMAL PARAMETERS")
                for param, value in step.best_tune_params.items():
                    overview_lines.append(f"│     {param:<20} {value}")
                overview_lines.append("│")
            
            # Restrictions section
            if step.restrictions:
                overview_lines.append("│  🚫 PARAMETER RESTRICTIONS")
                for restriction in step.restrictions:
                    # Wrap long restrictions nicely
                    if len(restriction) <= 60:
                        overview_lines.append(f"│     • {restriction}")
                    else:
                        # Split long restrictions into multiple lines
                        words = restriction.split()
                        current_line = "│     • "
                        for word in words:
                            if len(current_line + word) <= 76:
                                current_line += word + " "
                            else:
                                overview_lines.append(current_line.rstrip())
                                current_line = "│       " + word + " "
                        if current_line.strip() != "│":
                            overview_lines.append(current_line.rstrip())
                overview_lines.append("│")
            
            # Timestamp and completion
            overview_lines.append(f"│  📅 Recorded: {step.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Step separator (except for last step)
            if i < len(self.steps):
                overview_lines.append("└─────────────────────────────────────────────────────────────────────────────")
                overview_lines.append("")
            else:
                overview_lines.append("└─────────────────────────────────────────────────────────────────────────────")
        
        overview_lines.append("")
        
        # Final summary
        overview_lines.append("OPTIMIZATION COMPLETE")
        overview_lines.append("-" * 40)
        if self.baseline_time:
            overview_lines.append(f"🎯 Achieved {self.get_total_improvement():.2f}% performance improvement")
            overview_lines.append(f"🚀 Kernel is now {self.baseline_time / self.steps[-1].new_execution_time:.2f}x faster")
        else:
            overview_lines.append(f"✅ Optimization completed with final time: {self.steps[-1].new_execution_time:.6f}s")
        
        overview_lines.append("=" * 80)
        
        return "\n".join(overview_lines)
    
    def has_improvements(self) -> bool:
        """Check if any optimization steps have been recorded.
        
        Returns:
            True if at least one optimization step has been recorded, False otherwise.
        """
        return len(self.steps) > 0