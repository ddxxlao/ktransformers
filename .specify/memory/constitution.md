<!--
SYNC IMPACT REPORT
Version Change: Initial → 1.0.0
Modified Principles: N/A (initial creation)
Added Sections:
  - Core Principles (9 principles based on Unix Philosophy)
  - Performance & Architecture Standards
  - Development Workflow
  - Governance
Removed Sections: N/A
Templates Requiring Updates:
  ✅ .specify/templates/plan-template.md - reviewed, compatible
  ✅ .specify/templates/spec-template.md - reviewed, compatible
  ✅ .specify/templates/tasks-template.md - reviewed, compatible
  ✅ .specify/templates/agent-file-template.md - reviewed, compatible
Follow-up TODOs: None
-->

# KTransformers MoE Inference Framework Constitution

## Core Principles

### I. Single Responsibility & Modularity
**Rule**: Each component MUST do one thing well. Components are independently buildable, testable, and replaceable.

**Rationale**: The hybrid CPU+GPU scheduling system is inherently complex. Breaking it into focused, single-purpose modules (scheduler, cost model, cache manager, operator kernels) prevents cascading failures and enables parallel development.

**Requirements**:
- Each module has a clear, documented purpose stated in its header/README
- Module interfaces are minimal and cohesive (no "god objects")
- Dependencies are explicit and acyclic (DAG structure enforced)
- A component failing its tests MUST NOT block unrelated components from building

### II. Clean Interfaces & Composition
**Rule**: Components communicate through well-defined, stable interfaces. Prefer composition over monolithic designs.

**Rationale**: Expert scheduling, kernel execution, and cache management evolve at different rates. Clean interfaces allow independent optimization and testing.

**Requirements**:
- All inter-module boundaries defined via header files (.h/.hpp) or abstract base classes
- Interface contracts documented with preconditions, postconditions, invariants
- Implementation details hidden behind interfaces (no leaky abstractions)
- Modules testable in isolation via mock implementations of dependencies

### III. Text-Based Observability
**Rule**: System state, metrics, and debugging information MUST be exportable as structured text (JSON, CSV, plain text logs).

**Rationale**: Debugging hybrid CPU+GPU scheduling requires inspecting PCIe transfer timings, cache hit rates, and scheduling decisions. Text formats enable standard Unix tools (grep, awk, jq) for analysis.

**Requirements**:
- All cost model calculations loggable as JSON or CSV
- Scheduler decisions (which expert → CPU/GPU) emitted to stdout/file in structured format
- Performance metrics (latency, throughput, cache hit rate) exportable to Prometheus text format or equivalent
- Error messages written to stderr with context (timestamp, module, severity)

### IV. Early & Iterative Validation
**Rule**: Build and test components early. Discard implementations that don't meet performance or simplicity goals.

**Rationale**: The cost model and scheduling heuristics are hypotheses. Early validation prevents investing in approaches that don't deliver.

**Requirements**:
- Benchmarks exist before optimization (no "premature optimization")
- Each Phase 1 component has a smoke test verifying basic functionality
- Performance regression tests gate commits (e.g., inference latency ≤ baseline)
- Failed experiments documented in research.md with lessons learned

### V. Standardized Data Formats
**Rule**: Components exchange data in standardized, documented formats. Prefer self-describing formats (JSON, Protobuf) over binary blobs.

**Rationale**: Expert weights, activation tensors, and scheduling metadata flow between CPU/GPU, Python/C++, and cache layers. Standardized formats prevent integration bugs.

**Requirements**:
- Tensor metadata includes shape, dtype, device, memory layout
- Serialized expert weights use GGUF or a documented custom format with version tags
- Configuration files (YAML/JSON) validated against published schemas
- Inter-process communication uses text protocols (JSON-RPC, HTTP) unless benchmarks justify binary

### VI. Test-Driven Development (NON-NEGOTIABLE)
**Rule**: Tests MUST be written before implementation. Tests MUST fail initially, then pass after implementation.

**Rationale**: Given the complexity of scheduling, PCIe transfers, and kernel fusion, untested code is assumed broken.

**Requirements**:
- Red-Green-Refactor cycle strictly enforced: Write test → Verify it fails → Implement → Verify it passes
- Code reviews MUST verify tests existed before implementation (check git history)
- Coverage target: ≥80% line coverage for critical paths (scheduler, cost model, cache eviction)
- No merging to main without passing tests

### VII. Contract & Integration Testing
**Rule**: Interfaces between major subsystems (Python ↔ C++, CPU ↔ GPU, scheduler ↔ cache) require contract tests. Integration tests validate end-to-end workflows.

**Rationale**: The system spans multiple languages, devices, and threading models. Contract tests catch ABI/API mismatches before integration.

**Requirements**:
- Contract tests for every public API boundary (e.g., Python bindings, scheduler API)
- Integration tests for critical paths: token generation latency, expert loading, cache eviction
- Contract changes require updating tests BEFORE changing implementation
- Integration test suite runs in CI on every PR

### VIII. Performance & Complexity Accountability
**Rule**: Optimizations MUST be justified with benchmarks. Complexity MUST be justified with rationale. YAGNI (You Aren't Gonna Need It) until proven necessary.

**Rationale**: The system's goal is minimal latency on consumer hardware. Unjustified complexity increases maintenance burden without proven benefit.

**Requirements**:
- Optimizations documented with before/after benchmarks (e.g., "reduced PCIe stalls by 30%")
- Non-obvious algorithms (e.g., greedy scheduling) include rationale and alternatives considered
- Rejected approaches documented in research.md to avoid revisiting
- Complexity budget: Prefer simple heuristics (e.g., greedy scheduling) over ML models unless benchmarks show ≥20% improvement

### IX. Portability & Tooling Leverage
**Rule**: Use existing, proven tools and libraries. Avoid reinventing standard functionality. Design for Linux first, accommodate other platforms via abstraction.

**Rationale**: The project builds on PyTorch, CUDA, and llama.cpp. Reinventing tensor ops or GGUF parsing wastes effort.

**Requirements**:
- Use PyTorch for tensor ops unless custom kernels justified by benchmarks
- Use llama.cpp GGUF loader unless incompatible with expert caching requirements
- Build system: CMake + Python setuptools (already in use)
- Shell scripts (bash) for automation, documented for Ubuntu 22/24
- Platform-specific code isolated behind interfaces (e.g., CUDA vs CPU kernels)

## Performance & Architecture Standards

### Latency Requirements
- **Target**: Per-token latency ≤ 2× pure GPU baseline for cached experts
- **Measurement**: p50, p95, p99 latencies logged per generation
- **Regression Gate**: PRs cannot increase p95 latency by >10% without justification

### Memory Management
- **GPU VRAM**: Treated as LRU/value-driven cache for experts
- **CPU Memory**: NUMA-aware allocation for expert weights (if NUMA enabled)
- **Leak Detection**: Valgrind or AddressSanitizer run on integration tests

### Concurrency Model
- **GPU Kernels**: CUDA streams for overlapping compute + PCIe transfers
- **CPU Threads**: Thread pool for parallel expert computation
- **Synchronization**: Lock-free data structures where benchmarks justify (e.g., scheduler work queue)

### Build Standards
- **Incremental Builds**: CMake + Ninja for <30s rebuild on code changes
- **Reproducibility**: Fixed CUDA arch (8.9), pinned dependency versions in requirements.txt
- **CI Build Time**: Full build from scratch ≤ 20 minutes

## Development Workflow

### Feature Development Lifecycle
1. **Specification** (spec.md): Define user-facing behavior, no implementation details
2. **Planning** (plan.md): Technical approach, architecture, constitution compliance check
3. **Research** (research.md): Evaluate alternatives, document trade-offs
4. **Design** (contracts/, data-model.md): Define interfaces, data structures
5. **Task Breakdown** (tasks.md): Granular tasks with dependencies, parallel execution plan
6. **TDD Implementation**: Write tests → Verify failure → Implement → Verify pass
7. **Integration**: End-to-end validation, performance benchmarks
8. **Documentation**: Update README, API docs, quickstart guide

### Code Review Requirements
- **Constitution Check**: Reviewer verifies principles I-IX compliance
- **Test Verification**: Check git history confirms tests predated implementation
- **Benchmark Requirement**: Performance-critical changes include before/after metrics
- **Complexity Justification**: Non-obvious code requires comment explaining rationale

### Quality Gates (CI)
- All tests pass (contract, integration, unit)
- No performance regressions (latency, memory)
- Build succeeds on clean Ubuntu 22/24 environment
- Linting passes (if configured)

### Documentation Standards
- **README.md**: Quick start, build instructions, project overview
- **Architecture Docs**: High-level diagrams (scheduler, cache, data flow)
- **API Contracts**: Function signatures with preconditions/postconditions
- **Benchmarks**: Recorded in docs/ or embedded as code comments

## Governance

### Amendment Procedure
1. Propose change via PR to this constitution
2. Document rationale: What problem does this solve? What's the cost?
3. Impact analysis: Which templates, workflows, or code require updates?
4. Approval: Project maintainers vote (majority approval required)
5. Migration plan: Update affected specs, plans, tasks, code
6. Version bump: MAJOR for principle removal/redefinition, MINOR for additions, PATCH for clarifications

### Versioning Policy
- **MAJOR.MINOR.PATCH** semantic versioning
- **MAJOR**: Backward-incompatible principle changes (e.g., dropping TDD requirement)
- **MINOR**: New principles or sections added (e.g., adding security principle)
- **PATCH**: Clarifications, typo fixes, non-semantic edits

### Compliance Enforcement
- All PRs reviewed against constitution principles
- Complexity must be justified with rationale and benchmarks
- Violations require either code refactor or constitution amendment
- Constitution supersedes undocumented team practices

### Runtime Development Guidance
For agent-specific development guidance during implementation, refer to:
- `.specify/templates/agent-file-template.md` for AI agent instructions
- `.github/prompts/constitution.prompt.md` for constitution update procedures
- Project README.md and docs/ for technical onboarding

**Version**: 1.0.0 | **Ratified**: 2025-10-04 | **Last Amended**: 2025-10-04