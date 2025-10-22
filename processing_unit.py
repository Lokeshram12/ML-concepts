import numpy as np
import time

# Create large arrays
a = np.random.rand(10_000_000_0)
b = np.random.rand(10_000_000_0)

start = time.time()
c = a + b
end = time.time()

print("CPU Time:", end - start, "seconds")


import cupy as cp
import time

a = cp.random.rand(10_000_000)
b = cp.random.rand(10_000_000)

cp.cuda.Stream.null.synchronize()  # Ensure GPU is ready
start = time.time()
c = a + b
cp.cuda.Stream.null.synchronize()  # Wait until done
end = time.time()

print("GPU Time:", end - start, "seconds")

