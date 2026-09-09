# J.A.R.V.I.S Adaptive Recovery Evaluation Report

**Generated:** 2026-09-09 22:23:37  
**Overall Recovery Success Rate:** 100.0%  
**Average Recovery Duration:** 0.0028 s  
**Average Waves per Incident:** 1.5  
**Skipped LLM Invocations via Heuristics:** 3  
**Deterministic Early Exits:** 3  
**False Heuristic Exits:** 0 (Zero false rejections)  

---

## Scenario-by-Scenario Evaluation

| Scenario | Outcome | Waves | Duration (s) | Replanned | Skipped LLM Calls | Early Exit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `transient_failure` | **PASS** | 2 | 0.0034s | True | 0 | False |
| `multiple_failures` | **PASS** | 2 | 0.0106s | True | 0 | False |
| `cascading_dependency_failures` | **PASS** | 2 | 0.0028s | True | 0 | False |
| `provider_failure_classification` | **PASS** | 1 | 0.001s | False | 0 | False |
| `timeout_failure_classification` | **PASS** | 1 | 0.001s | False | 0 | False |
| `permanent_validation_failure` | **PASS** | 1 | 0.001s | False | 1 | True |
| `impossible_recovery_retry_budget` | **PASS** | 3 | 0.001s | False | 1 | True |
| `repeated_identical_recovery` | **PASS** | 0 | 0.0019s | True | 1 | True |

---

## Invariant Validation

- **Zero False Heuristic Exits**: RecoveryHeuristics never aborted an otherwise recoverable workflow.
- **Cascading Integrity**: Upstream failure properly isolated descendants and resumed execution post-recovery.
- **Infinite Loop Defense**: Repeated identical recovery plans were immediately detected and halted.
- **Fast Rejection**: Permanent validation errors short-circuit in < 1 ms without invoking LLM providers.
