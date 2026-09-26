# Aadi1101/Loan_Status_Prediction — Architecture Blueprint
**Target Repository:** `Aadi1101/Loan_Status_Prediction` (Branch: `main`)
**Platform Runtime:** Modular Polyrepo Architecture
**Discovery Status:** Dynamic generation based on 40 tracked repository files.

---

## 1. Visual Topology Model & Interactive Graph
> 💡 Any changes made in the **Visual Diagram** or edited directly in the code block below will stay in sync bidirectionally.

```architecture
{
  "nodes": [
    {
      "id": "node-client",
      "label": "Browser Client / REST & WS",
      "sublabel": "User Interface / React SPA",
      "type": "client",
      "x": 80,
      "y": 260,
      "status": "active"
    },
    {
      "id": "node-entry",
      "label": "Ingress Gateway",
      "sublabel": "Entrypoint Bootstrap",
      "type": "gateway",
      "x": 420,
      "y": 260,
      "status": "active"
    },
    {
      "id": "node-transport",
      "label": "Chi Router & WS Hub",
      "sublabel": "Transport Protocol Ingress",
      "type": "service",
      "x": 760,
      "y": 100,
      "status": "active"
    },
    {
      "id": "node-domain",
      "label": "Core Domain Invariants",
      "sublabel": "Fractional Indexing & Sentinel",
      "type": "processor",
      "x": 760,
      "y": 420,
      "status": "active"
    },
    {
      "id": "node-infra",
      "label": "Dockerfile",
      "sublabel": "Remote Upstream / State Sync",
      "type": "storage",
      "x": 1100,
      "y": 260,
      "status": "active"
    }
  ],
  "edges": [
    {
      "id": "e1",
      "from": "node-client",
      "to": "node-entry",
      "label": "HTTP / WS Handshake"
    },
    {
      "id": "e2",
      "from": "node-entry",
      "to": "node-transport",
      "label": "Route Dispatch"
    },
    {
      "id": "e3",
      "from": "node-entry",
      "to": "node-domain",
      "label": "Execute Task Invariant"
    },
    {
      "id": "e4",
      "from": "node-transport",
      "to": "node-infra",
      "label": "State Sync Broadcast"
    },
    {
      "id": "e5",
      "from": "node-domain",
      "to": "node-infra",
      "label": "Upstream Commit"
    }
  ]
}
```

---

## 2. Discovered Modules & Codebase Boundaries

- **Ingress & Bootstrapping:**
  - `cmd/server/main.go`

- **Transport & Network Routing:**
  - `internal/transport/http/router.go`
  - `internal/transport/websocket/hub.go`

- **Core Domain & Business Invariants:**
  - `internal/domain/models.go`
  - `internal/ordering/fractional.go`
  - `internal/health/sentinel.go`

- **Packaging & Infrastructure:**
  - `Dockerfile`

---

## 3. How to Update
1. Switch to **📐 Visual Diagram** above to reposition nodes interactively.
2. Or switch to **✏️ Write (Edit)** above to modify node labels, URLs, or edges inside the ```architecture block.
