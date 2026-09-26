# Technical Design Specification
**Service:** Aadi1101/Loan_Status_Prediction

---

## 1. Data Contracts & Schema
- **Issue Entity:** `{ id, projectKey, issueKey, title, status, orderKey, storyPoints, assignee }`
- **Fractional Index Engine:** Generates midpoints between predecessor `prevKey` and successor `nextKey` using ASCII Base-62 characters (`0-9`, `A-Z`, `a-z`).

## 2. API Endpoints
- `GET  /api/v1/issues?projectKey=:key` — Fetch workspace issues
- `POST /api/v1/issues/reorder` — Reorder card via fractional order key
- `POST /api/v1/health/autofix` — Run autonomous invariant sentinel
- `GET  /api/v1/github/tree` — Recursive repository file tree
- `GET  /api/v1/github/file` — Raw file content inspector
- `GET  /ws` — Duplex WebSocket streaming connection