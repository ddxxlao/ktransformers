# **Requirement Specification: Performance Profiling and Bottleneck Analysis for KTransformer**

## **1. Objective**

To build a comprehensive profiling and analysis toolkit integrated with the KTransformer framework. The primary goal is to precisely measure the performance of each component during MoE model inference, identify bottlenecks, and collect the empirical data needed to design and validate a hybrid CPU+GPU scheduler.

## **2. Scope of Analysis**

*   **Framework:** KTransformer Inference Framework.
*   **Inference Phase:** The analysis will focus exclusively on the **autoregressive decode phase**, as this is where memory-bound operations become the primary bottleneck. The prefill phase is out of scope for this analysis.
*   **Decode Strategy:** For now, all token and experts is calculated on CPU, without any offloading to GPU. This will serve as the baseline for future comparisons.
*   **Hardware Focus:** The system must capture metrics related to the CPU, GPU, system DRAM, and the PCIe bus.
*   **Model:** The initial target model for profiling will be Qwen3-30B-A3B, but the solution should be general enough for other MoE models.

## **3. Core Metrics to Capture**

The profiling system must be able to measure and log the following metrics with microsecond-level precision where possible.

**A. End-to-End Latency:**
*   `T_token`: Total time to generate a single token during the decode phase. This is the primary top-level metric to be minimized.

**B. Granular Component Timings (Latency Breakdown):**
*   `T_routing`: Time taken by the gating network to select the top-k experts.
*   `T_scheduling`: Time taken by the (future) scheduler to decide the execution plan (CPU vs. GPU allocation). (Will Implement in the future)
*   `T_cpu_computation`: The time the CPU spends executing an expert's forward pass (`T_cpu_comp`).
*   `T_sync`: Time spent on synchronization primitives (e.g., waiting for a PCIe copy to complete, waiting for a GPU kernel to finish). This represents idle/stall time.

Will Need to Collect after Implementing GPU Offloading, for now we just use CPU to compute all experts:
*   `T_pcie_transfer`: The pure data transfer time for moving a single expert from CPU RAM to GPU VRAM. This should be measured independently of computation. (Will Implement in the future)
    *   This can be approximated as `S_exp / B_pcie`, where `S_exp` is the size of one expert and `B_pcie` is the effective PCIe bandwidth measured during the run.
*   `T_gpu_kernel_launch`: Overhead associated with launching a CUDA kernel.
*   `T_gpu_computation`: The actual time the GPU spends executing an expert's forward pass (`T_gpu_comp`).

**C. Hardware Utilization Metrics:**
*   **GPU Utilization:**
    *   Tensor Core Activity (%).
    *   Overall SM (Streaming Multiprocessor) Utilization (%).
    *   Memory Bandwidth Utilization (GB/s).
    *   VRAM Usage (GB).
*   **CPU Utilization:**
    *   Per-core utilization (%) to see if the workload is single-threaded or multi-threaded.
    *   CPU cache miss rates (L1/L2/L3).
*   **PCIe Bus:**
    *   Effective Bandwidth (GB/s) during expert transfers.
    *   Total data volume transferred per token.

## **4. Functional Requirements & Implementation Plan**

**FR1: Code Instrumentation**
*   **Requirement:** Integrate high-precision timers directly into the KTransformer C++/Python source code.

**FR2: External Profiler Integration**
*   **Requirement:** Create scripts and configurations to run KTransformer under industry-standard profilers.

**FR3: Data Logging and Aggregation**
*   **Requirement:** All collected metrics (from in-code timers and external tools) must be saved in a structured, machine-readable format. (e.g., JSON, CSV).

**FR4: Data Visualization and Reporting**
*   **Requirement:** Develop scripts to parse the log files and generate human-readable reports and visualizations.

## **5. Key Questions to Answer with the Analysis**

The final analysis report produced by this system should definitively answer:

1.  What is the average end-to-end latency per token, and what is its variance?
2.  What is the single biggest contributor to this latency? (e.g., `T_pcie_transfer`)
3.  What is the empirically measured `T_pcie_transfer` for one expert on your hardware? Does it match the theoretical `S_exp / B_pcie`?
4.  How much time is the GPU idle while waiting for data?
5.  How do `T_cpu_comp` and `T_gpu_comp` compare? Is the GPU faster, and by how much?
6.  Are the CPU cores saturated during expert computation, or are they waiting on memory?
7.  Is there any overlap between CPU computation, GPU computation, and PCIe transfers in the current baseline?

## **6. Deliverables**

1.  **Instrumented Code:** A branch of the KTransformer repository with the added in-code profiling hooks.
2.  **Profiling Scripts:** A set of shell scripts to automate the process of running inference under `nsys` and `perf`.
3.  **Analysis Scripts:** A Python script (`analyze.py`) that takes the raw data logs and generates the final report and plots.
4.  **Baseline Performance Report:** A detailed document (e.g., a Markdown file) presenting the answers to the key questions above, supported by the generated plots and data. This report will serve as the benchmark against which all future optimizations will be measured. Which will as the only CPU baseline for future comparison.
5.  **Documentation:** A README file explaining how to use the profiling tools and interpret the results.