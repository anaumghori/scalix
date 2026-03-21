# GPU execution model

| Concept | Explanation |
|---|---|
| **Streaming Multiprocessors** | A modern NVIDIA GPU is composed of many identical compute units called **streaming multiprocessors**. An SM is the fundamental scheduling and execution unit of the GPU. Each SM contains several smaller arithmetic units, scheduling logic, register files, shared memory, and control hardware. From the perspective of program execution, the SM is the unit that receives groups of threads and executes their instructions. |
| **CUDA cores** | Inside each SM are many **streaming processors**, commonly called **CUDA cores**. A CUDA core is a scalar arithmetic execution unit capable of performing operations such as floating-point addition, multiplication, or fused multiply–add. If the SM is viewed as a small processor cluster, the CUDA cores are the individual arithmetic lanes inside that cluster. |
| **Threads** | The GPU programming model exposes parallelism through **threads**. A thread represents a single logical execution context with its own registers and program counter. In a typical GPU program thousands or millions of threads are created. Each thread executes the same program but operates on different data elements. |
| **Warps** | Although the programmer writes code in terms of individual threads, the hardware schedules threads in fixed groups called **warps**. A warp consists of 32 threads. These 32 threads execute the same instruction at the same time in a lockstep fashion. This model is often called **SIMT** (single instruction, multiple threads). When the GPU issues an instruction, it does so for an entire warp. If all threads in the warp follow the same control path, execution proceeds efficiently. If threads diverge because of conditional branches, the hardware serializes the execution paths internally. |
| **Thread blocks** | Threads are organized into larger groups called **thread blocks**. A thread block is a set of threads that execute on the same SM and can cooperate through shared memory and synchronization barriers. Because all threads in a block reside on a single SM, they can communicate efficiently using the on-chip memory available within that SM. |
| **Kernel** | At the highest level is the **kernel**. A kernel is a function that runs on the GPU and is launched from the CPU. When a kernel is launched, the programmer specifies a grid of thread blocks. Each block contains a certain number of threads, and the GPU distributes those blocks across the available SMs. |
| **Overall workflow** | A kernel launch creates a grid of thread blocks. Each thread block contains many threads. Threads are scheduled in groups of 32 called warps. Warps execute on streaming multiprocessors. Inside each SM, CUDA cores perform the actual arithmetic instructions for those threads. This hierarchy exists because GPUs aim to expose extremely large amounts of parallelism. The programmer describes the parallel work as threads and blocks, while the hardware schedules those threads onto SMs and executes them using warps and CUDA cores. |
| **Specialized compute units** | Hardware units designed to accelerate specific types of computation beyond the capabilities of general-purpose CUDA cores, implementing optimized pipelines for operations commonly used in certain workloads. For example: RT cores in NVIDIA GPUs accelerate ray tracing |
| **Tensor cores** | One of the specialized compute units, designed for high-throughput matrix-multiply-accumulate (MMA) operations. For example, it can take two small matrices, multiply them and then add the result to another matrix in a single operation. They also support mixed-precision arithmetic, meaning they can take inputs in lower-precision formats (like FP16 or even INT8) and accumulate results in higher precision (like FP32). |

### Dumb question I had: If threads are executed in warps of 32, why do we organize them into thread blocks?

Let’s define the hierarchy in two different layers: a **programming abstraction defined by the CUDA model** and an **execution grouping used by the hardware scheduler**.

1. A CUDA program launches a **grid**. The grid contains **thread blocks**, and each block contains **threads (Grid → Thread Blocks → Threads)**. This hierarchy is part of the programming model and is visible in the code. Each thread executes the kernel independently. The programmer specifies the number of threads per block and the number of blocks in the grid. A thread block might contain for example 64, 128, 256, or 1024 threads depending on the kernel design and hardware limits. The scheduler guarantees that all threads of a block execute on the **same SM**, which enables fast communication through shared memory.

2. The GPU hardware does not schedule individual threads. Instead it schedules **warps**, which are fixed groups of 32 threads. A warp is therefore an execution unit created internally by the hardware from the threads of a thread block. The practical hierarchy therefore becomes: Grid → Thread Blocks → Warps → Threads.

Warps are for **execution scheduling,** Thread blocks are for **program structure, memory sharing, and synchronization.** A warp always contains 32 threads. If a thread block contains N threads, the hardware divides them into : number of warps in a block = N/32

For example: 64 threads per block → 2 warps, 128 threads per block → 4 warps.


### Host code and kernel execution
Host code runs on the CPU and is responsible for coordinating and managing GPU execution. Its tasks include allocating memory on the GPU, transferring input data from CPU memory to GPU memory, launching kernels, and retrieving results after computation. The host determines when kernels are launched and specifies the execution configuration, such as the number of threads per block and the number of blocks in a grid. Device code, or the kernel, runs on the GPU and performs parallel computation across many threads. Once the host launches the kernel, the GPU executes it on its streaming multiprocessors (SMs), performing the computation efficiently in parallel. This division exists because GPUs are highly optimized for parallel numerical computation, whereas CPUs handle general program control, system interactions, and coordination tasks that GPUs are not designed to perform.


### Thread coarsening and warp stalls
If a thread in a warp encounters a long-latency operation, such as a memory load from global memory, the entire warp may stall while waiting for that instruction to complete. Similarly, if the number of active threads is small relative to the hardware capacity, the GPU cannot fully hide memory or instruction latency, reducing throughput. **Thread coarsening** addresses this by combining the work of multiple logical threads into a single physical thread. Instead of having many fine-grained threads that each perform a very small operation, each thread performs a larger chunk of computation. This increases instruction-level parallelism within each thread and reduces the total number of active threads required to process the workload.  

There are several consequences. First, coarsening can improve warp occupancy by ensuring that each thread performs more useful work per scheduling unit. Second, it can reduce warp stalls caused by memory latency or control divergence, because fewer total threads are required and each thread can perform multiple dependent operations without needing to wait for other threads. Coarsening is most useful in situations where the workload per thread is very light, the kernel is memory-bound, or the computation per element is small relative to the cost of scheduling a warp.  

However, thread coarsening has trade-offs. It increases register and shared memory usage per thread. Excessive coarsening can reduce the total number of threads that can reside on an SM, lowering occupancy and potentially reducing performance if the hardware cannot hide latency effectively.


### Minimizing control divergence
Control divergence occurs when threads in a single warp follow different execution paths. GPUs operate on the SIMT (single instruction, multiple threads) model, meaning all threads in a warp execute the same instruction at any given cycle. When threads take different branches in a conditional statement, such as an `if` statement, the hardware cannot execute both paths simultaneously within the warp. Instead, the warp is serialized: threads taking one path are executed first while threads taking the other path are masked off, then the other path executes, and the results are merged. This serialization effectively reduces throughput and introduces inefficiency. Control divergence happens naturally in kernels that contain conditional statements dependent on per-thread data. The more threads in a warp that diverge from the common path, the greater the performance penalty, because fewer threads execute simultaneously per cycle.  

To reduce divergence, kernel designers aim to align the control flow of threads within a warp. Techniques include sorting or partitioning data so that threads with similar execution paths reside in the same warp, restructuring algorithms to reduce branching, or using predication (executing both branches but masking the results conditionally) when the alternative paths are short. The goal is to maintain lockstep execution as much as possible so that all 32 threads in a warp remain active on the same instruction, maximizing throughput.

<br><br>


# GPU memory hierarchy

| Concept | Explanation |
|---|---|
| **Register file** | The smallest and fastest memory in the GPU is the **register file**. Registers store the private variables of a thread during execution. Every thread has its own set of registers, and they are physically located inside the SM that executes that thread. Accessing a register takes only a few clock cycles because the data resides directly in the execution unit’s register file. Each SM has a fixed number of physical registers. If a kernel requires too many registers per thread, the GPU can run fewer threads simultaneously on each SM. This relationship between register usage and concurrency is one of the key performance considerations. For example, assume an SM contains 65536 registers. If a kernel uses 32 registers per thread, then 65536/32 = 2048. Up to 2048 threads could theoretically reside on that SM. |
| **Shared Memory** | The next level in the hierarchy is **shared memory**. Shared memory is a manually managed memory region located on each SM. All threads within a block can access it. It is much faster than global memory because it resides on the chip near the compute units. Access latency is typically tens of cycles rather than hundreds. Shared memory is often used to store data that multiple threads in the block will reuse. Because threads can cooperate through shared memory, algorithms can load data once from global memory and then reuse it many times locally. |
| **L1 cache** | Closely related to shared memory is the **L1 cache**. On many architectures, the shared memory and L1 cache share the same on-chip memory pool. The L1 cache automatically stores recently accessed global memory data to reduce latency for repeated accesses. |
| **L2 cache** | Above the SM level sits the **L2 cache**. The L2 cache is shared by all streaming multiprocessors on the GPU. Whenever an SM requests data from global memory, the request passes through the L2 cache. If the data is present there, the request can be satisfied quickly without accessing external memory. |
| **Global Memory** | The largest level of memory is **global memory**. This memory resides in external DRAM chips attached to the GPU. For example, a GPU such as the NVIDIA H100 contains tens of gigabytes of high-bandwidth memory. Global memory is accessible by all threads on the device and persists across kernel launches. However, global memory has significantly higher latency than on-chip memory. Accessing it may require hundreds of cycles. GPUs rely on massive parallelism and memory coalescing to hide this latency. |
| **Overall workflow** | During execution, data typically flows from global memory into the SM through the L2 cache and L1 cache. Once loaded, it may be stored in registers or shared memory for computation. After computation finishes, results may be written back through the same path to global memory. |
| **Tiling** | **Tiling** is a technique that reorganizes computations so that small blocks of data are repeatedly reused from fast memory rather than repeatedly loaded from global memory. Consider computing C = A × B for large matrices. Each element of C requires a dot product of a row of AAand a column of B. A naive implementation would repeatedly read the same rows and columns from global memory for many different threads. Because global memory is slow, this leads to inefficient execution. Tiling reorganizes the computation so that small submatrices (tiles) of A and B are loaded into shared memory once. Threads within the block then perform many arithmetic operations using those tiles before loading new data. |

### DRAM and burst memory behavior

Global memory on the GPU is implemented using **DRAM** (dynamic random access memory). DRAM stores bits as electrical charges in capacitors arranged in large two-dimensional arrays. Each DRAM chip contains rows and columns of memory cells. When a memory address is requested, the DRAM hardware activates an entire row of memory cells simultaneously. This row is transferred into a buffer called a **row buffer**.

After this step, the memory system still needs to choose **which part of the row** should be sent to the requester. That selection is what is called a **column access**. Consecutive columns within that row can be read extremely quickly.

**Burst transfer:** It’s a mechanism where instead of returning only the single requested memory location, the DRAM system can deliver a contiguous block of nearby memory values in one high-bandwidth transfer. This is possible due to the multiple reads to different columns within the same active row.

Two important cases.

1. A **row hit** occurs when the requested address lies in the row that is already active in the row buffer. The memory controller can directly perform a column access, which is fast.
2. A **row miss** occurs when the requested address belongs to a different row. The controller must precharge the bank, close the existing row, and activate a new row. This takes significantly more time.

### Memory coalescing

Memory coalescing is a mechanism that combines memory requests from multiple threads in a warp into a small number of efficient transactions. Recall that a warp contains 32 threads. Suppose each thread requests a different element from global memory. If those elements lie in consecutive addresses, the hardware can combine the 32 requests into one or a few burst accesses to DRAM. Because these addresses are contiguous, the memory system can fetch them using a single burst from DRAM. The hardware then distributes the retrieved data to the individual threads.

If the threads access scattered addresses, the GPU must perform multiple independent memory transactions. This dramatically reduces effective bandwidth. Memory coalescing therefore aligns the execution behavior of warps with the burst-oriented nature of DRAM.
