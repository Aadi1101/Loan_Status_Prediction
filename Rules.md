# Aadi1101/Loan_Status_Prediction — Platform & Codebase Invariants
**Status:** ACTIVE & ENFORCED

---

## 1. Architectural & Concurrency Rules
- `INV-01 (Zero-Lock Card Ordering)`: Drag-and-drop operations MUST compute lexicographical fractional midpoints without locking database tables.
- `INV-02 (Constant-Time Verification)`: Cryptographic signature checks for incoming GitHub webhooks must use constant-time comparisons (`crypto/subtle`).
- `INV-03 (Strict Project Scoping)`: The Kanban Board and Backlog views must filter issues strictly by `projectKey`. No cross-workspace state leakage permitted.


## 2. Code Quality & Linter Conventions
- All Go files must pass `golangci-lint run --timeout=3m` with zero warnings.
- Commits pushed to `main` must pass automated unit and race tests (`go test -race ./...`).
- WebSocket event payloads must conform to unified envelope `{ "type": string, "data": any }`.
