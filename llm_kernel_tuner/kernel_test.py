import numpy as np
from typing import Tuple, Union, List, Optional
from numpy.typing import NDArray

ArrayOfIntOrFloat32 = Union[NDArray[np.int32], NDArray[np.float32]]
ScalarIntOrFloat32 = Union[np.int32, np.float32]
# only these types are allowed for kernel tuner
TestInputType = List[Union[ArrayOfIntOrFloat32, ScalarIntOrFloat32]]


class KernelTest:
    """
    Kernel test that will test the correctness of the kernel being tuned.
    
    Args:
        input_data (np.ndarray): Input array for kernel testing. Should be a 2D array
            where the first dimension represents different parameter sets and the 
            second dimension contains the values for each parameter.
        expected_output (np.ndarray): Expected kernel output for validation. Should be a 2D array
            with the same first dimension as ``input_data``. Elements with None value will be
            excluded from comparison with actual kernel output.
        size (int | Tuple[int, ...]): Problem size specification used by the tuning 
            process. May be a single integer or a tuple of dimensions depending on the 
            kernel requirements.
    """
    def __init__(self, input_data: List[TestInputType], expected_output: List[Optional[TestInputType]], size: Union[int, Tuple[int, ...]]):
        self.input_data = input_data
        self.expected_output = expected_output
        self.size = size

    def __repr__(self):
        return f"KernelTest(input_data={self.input_data}, expected_output={self.expected_output}, size={self.size})"