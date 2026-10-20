# DocuMind AI — RAG Evaluation & Benchmark Methodology

This document outlines the evaluation framework, metric formulations, benchmark dataset specifications, and reliability criteria used by **DocuMind AI**.

---

## 1. Evaluation Philosophy

In high-stakes enterprise applications, Retrieval-Augmented Generation (RAG) cannot be evaluated solely on generated text fluency. A system must be measured across two independent, decoupled operational axes:

1. **Retrieval Efficacy**: Did the vector store and keyword indexes return the authentic, relevant document chunks containing the necessary factual evidence?
2. **Generation Grounding & Attribution**: Did the model synthesize an answer strictly backed by the provided evidence, accurately cite 1-based source pages, and refuse to answer when evidence was absent?

---

## 2. Quantitative Metric Formulations

### 2.1 Hit Rate @ K
- **Definition**: Binary indicator determining whether at least one relevant passage containing gold evidence appears within the top $K$ retrieved candidates.
- **Formula**:
  $$\text{Hit@K} = \begin{cases} 1 & \text{if } \exists r \in \text{top}_K(R) \text{ s.t. } \text{is\_relevant}(r) \\ 0 & \text{otherwise} \end{cases}$$
- **Interpretation**: Measures retrieval coverage. A target of $\ge 90\%$ ensures the LLM is provided with relevant evidence in almost all queries.

### 2.2 Mean Reciprocal Rank (MRR)
- **Definition**: Evaluates the ranking position of the first relevant candidate passage.
- **Formula**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  where $\text{rank}_i$ is the rank position of the highest-ranked gold passage for query $i$.
- **Interpretation**: Higher MRR indicates that gold evidence is concentrated at the very top of the prompt context, reducing lost-in-the-middle context degradation.

### 2.3 Precision @ K & Recall @ K
- **Precision @ K**: Proportion of retrieved chunks in top $K$ that are authentic and relevant.
  $$\text{Precision@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$
- **Recall @ K**: Proportion of available relevant passages captured within the top $K$.
  $$\text{Recall@K} = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

### 2.4 Citation Grounding Precision
- **Definition**: The proportion of generated citation tags (`[Source 1]`, `[Source 2]`) that correspond to authenticated, retrieved context chunks and correctly reflect the 1-based source document and page number.
- **Formula**:
  $$\text{CitationPrecision} = \frac{\sum_{c \in C} \mathbf{1}_{\text{valid}}(c)}{|C|}$$
- **Interpretation**: Prevents hallucinated source references and phantom page numbers. Scores below $1.0$ indicate phantom citations.

### 2.5 Factual Grounding & Abstention Accuracy
- **Definition**: For answerable queries, measures whether mandatory factual anchors (numbers, entities, dates) are preserved in the response. For unanswerable queries, checks whether the model explicitly abstains.
- **Interpretation**: Ensures the application does not fabricate answers when document evidence is missing or out-of-domain.

---

## 3. Curated Benchmark Dataset (8 Test Scenarios)

The benchmark collection in `app/evaluation/benchmark_dataset.py` covers 8 multi-scenario test categories:

| Case ID | Scenario Category | Target Document | Expected Page | Expected Behavior | Evaluation Objective |
|---|---|---|---|---|---|
| `TC-01` | Direct Factual | `business_financial_report.pdf` | Page 1 | Answer (28% revenue growth) | Direct single-fact extraction |
| `TC-02` | Exact Keyword | `technical_architecture_guide.pdf` | Page 1 | Answer (PyMuPDF) | Technical term exact matching |
| `TC-03` | Policy Compliance | `ai_governance_policy.pdf` | Page 1 | Answer (POL-SEC-2025-04) | Formal policy ID extraction |
| `TC-04` | Semantic Paraphrase | `technical_architecture_guide.pdf` | Page 1 | Answer (BM25 + Hybrid) | Conceptual paraphrase search |
| `TC-05` | Tabular Data | `business_financial_report.pdf` | Page 1 | Answer (28.1M & 33.8M R&D) | Multi-column table retrieval |
| `TC-06` | Multi-Document | `technical_architecture_guide.pdf` & `ai_governance_policy.pdf` | Page 1 | Answer (0.70 & Section 2.1) | Cross-document synthesis |
| `TC-07` | Unanswerable / Abstention | *None (Out-of-corpus)* | N/A | **Abstain** (Refusal) | Prevents hallucination |
| `TC-08` | Policy Mandate | `ai_governance_policy.pdf` | Page 1 | Answer (Session disconnect) | Clause-specific attribution |

---

## 4. Deterministic vs. LLM-as-a-Judge Evaluation

DocuMind AI prioritizes **deterministic, reproducible evaluation**:
- Deterministic checks run locally in milliseconds with zero external API calls and zero evaluation costs.
- They do not fluctuate due to model updates or non-deterministic sampling temperatures.
- Optional LLM-as-a-judge approaches (such as Ragas or TruLens) may be used for qualitative nuance, but are subject to external API costs, model drift, and judge hallucinations.
