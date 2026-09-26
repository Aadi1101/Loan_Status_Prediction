# Engineering Task Decomposition
**Sprint Schedule:** Sprint 14 (Active)

---

## Phase 1: Core Engine & Invariant Sentinel
- [x] Scaffold modular Go backend with Chi routing and WebSocket hub.
- [x] Implement fractional indexing midpoint algorithm (`internal/ordering/fractional.go`).
- [x] Configure autonomous invariant auditor and auto-repair handler (`/health/autofix`).

## Phase 2: GitHub Repository Gateway
- [x] Wire GitHub REST client for user profile, repo listing, and tree fetching.
- [x] Add repository file contents API with Base64 decoding.
- [x] Support direct commit and push via in-app modal.

## Phase 3: Living Docs & Architecture Synchronization
- [x] Implement bidirectional markdown persistence with Go backend.
- [x] Render repository-aware living documentation for all project files.
- [ ] Connect production deployment webhook triggers.