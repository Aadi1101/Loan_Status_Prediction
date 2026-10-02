# Architectural Decision Log (ADR & Memory)
**Workspace Context:** `Aadi1101/Loan_Status_Prediction`



## ADR-001: Lexicographical Fractional Indexing
- **Decision:** Use fractional string order keys (`a0`, `a1`, `a0V`) instead of integer ranks.
- **Rationale:** Integer ranks require shifting all subsequent rows on insert, causing row-lock contention. Fractional midpoints execute with single-row updates.

## ADR-002: Autonomous Auto-Heal Bug Ticket Invariant
- **Decision:** When the health sentinel discovers an invariant violation (such as cyclic blockers or WIP limit overflows), it repairs the graph and generates a closed Bug ticket directly in `Done`.
- **Rationale:** Preserves historical auditability without human intervention.

## ADR-003: Authoritative Git Tree Sourcing
- **Decision:** Derive the Project Structure and Living Docs dynamically from the GitHub recursive Git Tree API rather than static client-side presets.
