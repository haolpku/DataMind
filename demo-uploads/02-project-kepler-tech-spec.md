# Project Kepler — Tech Spec (Draft v0.3)

**Project codename**: Kepler
**Owner**: Bob (Senior Engineer, Search Platform)
**Reviewers**: Ann (EM), 林涛 (AI Platform EM)
**Target quarter**: 2026-Q3
**Depends on**: Search Multi-Tenant Isolation (2026-Q2 in progress)

## 1. Motivation

Today's Search Platform supports text-only retrieval. Customers in industries like
manufacturing, real estate, and design have repeatedly asked for **image and PDF
search** — the ability to drop in a product photo, an engineering drawing, or a
scanned spec sheet, and get back the most relevant matches from their knowledge
base.

Project Kepler delivers this capability as a Q3 launch.

## 2. Scope

### In scope
- Image search via CLIP-style embeddings
- PDF parsing (text + page images)
- Mixed-modality reranking (text + image score fusion)
- API v3 endpoint: `POST /v3/search/multimodal`

### Out of scope
- Video search (Q4 candidate)
- 3D model search (no customer demand yet)
- Real-time camera streams (latency too high for current infra)

## 3. Architecture

### 3.1 Embedding pipeline
- New service: `embedding-multimodal`
- Models: CLIP-ViT-Large (default) + customer-specific finetunes
- Throughput target: 200 images/sec per GPU, 50 PDFs/sec per CPU

### 3.2 Storage
- Reuse Chroma cluster (sharding upgrade in Q2 unblocks this)
- New collection naming: `kb_<tenant>_visual` alongside `kb_<tenant>_text`

### 3.3 Retrieval
- Hybrid score = α·text_score + β·image_score + γ·rerank_score
- α/β/γ tunable per tenant (default 0.5/0.3/0.2)

## 4. Dependencies

| Dependency | Owner | Required by | Risk |
|---|---|---|---|
| Search Multi-Tenant Isolation | Ann | 2026-06-30 | High — blocks Kepler launch if delayed |
| Chroma Sharding Upgrade | Bob | 2026-05-31 | Low — already on track |
| Object Storage Quota Approval | Frank (Data Platform) | 2026-Q2 end | Medium |
| GPU Procurement | 王明 (Infra) | 2026-Q2 end | High — supply chain |

## 5. Team & Staffing

- **Tech Lead**: Bob (50% allocation)
- **Engineers**: 陈诚 (50%), Dana (50% Q3 onwards), 1 new hire (TBD)
- **Product Partner**: Ivy
- **Designer**: Karen (15% — for admin console UI)
- **Data Engineering**: Frank (consult basis)

## 6. Timeline

| Milestone | Date | Owner |
|---|---|---|
| Tech spec approved | 2026-05-15 | Ann |
| Embedding service alpha | 2026-06-30 | Bob |
| Internal dogfood | 2026-07-31 | 陈诚 |
| Beta with 3 design partners | 2026-08-15 | Ivy |
| GA launch | 2026-09-30 | Ann |

## 7. Risks & Mitigations

- **R1: Multi-tenant isolation slip** → Have fallback plan to launch Kepler on single-tenant first
- **R2: CLIP model performance on Chinese product images** → Pre-train domain adaptation in Q2
- **R3: GPU shortage** → Already engaged 王明 to lock 8 H100s by 2026-06
- **R4: Cross-team coordination overhead** → Weekly sync chaired by Bob, 30 min, Wednesdays 11:00

## 8. Open Questions

1. Should we also bundle existing OCR results into the multimodal index? (Pending decision from 林涛)
2. Pricing model — flat per-image vs tiered? (Product calls)
3. Rollout strategy — opt-in vs auto-migrate existing customers? (Need legal input from Queenie)

---

*Last edited 2026-04-28 by Bob. Comments welcome on #project-kepler.*
