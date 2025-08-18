import numpy as np
from typing import List
import warnings
import pickle
from multiprocessing import shared_memory, resource_tracker
import sys

warnings.filterwarnings("ignore")

def execute_cuda_test() -> List[np.ndarray]:
    {generated_code}
    
    return input_data

if __name__ == '__main__':
    input_data = execute_cuda_test()

    serialized_data = pickle.dumps(input_data)
    data_size = len(serialized_data)

    # 1. Send data size to parent
    print(data_size, flush=True)

    # 2. Wait for parent to send the shared memory name
    shm_name = sys.stdin.readline().strip()

    shm = None
    
    try:
        # 3. Attach to the shared memory block created by the parent
        shm = shared_memory.SharedMemory(name=shm_name)
        
        # 4. Write data to shared memory
        shm.buf[:data_size] = serialized_data
        
        # 5. Signal completion to parent
        print("DONE", flush=True)
    finally:
        if shm:
            resource_tracker.unregister(shm.name, 'shared_memory')
            shm.close()