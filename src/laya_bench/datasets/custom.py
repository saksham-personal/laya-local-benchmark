"""Deterministic company research orchestration decisions.

Every gold label comes from a hand-written counterfactual pair. Neither Laya
nor another model creates labels. The eight pair families per task are split
before their four surface variants are produced.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

DEFAULT_SEED = 20261005

_SCENARIOS = (
    "A user is in the local company-screening workspace. A saved database and prior assessments are available. Request:",
    "The orchestrator can use local structured fields, semantic company search, saved evidence, and external research. Request:",
    "An M&A analyst is reviewing company records. The request may refer to a current fact or a previously saved conclusion. Request:",
    "The application has company descriptions, financial fields, and a history of assessments. Decide based on the request:",
)

_FAMILIES: dict[str, dict[str, Any]] = {
    "intent": {
        "question": {"type": "choice", "instructions": "What is the user's primary intent?", "criteria": {
            "company_screening": "Find or shortlist companies matching investment or product criteria.",
            "company_comparison": "Compare two or more named companies or a saved shortlist.",
            "database_question": "Retrieve a fact or prior conclusion from stored company records.",
            "research_request": "Investigate current information from outside the stored workspace.",
            "report_generation": "Create or export a report, spreadsheet, or presentation.",
            "general_chat": "A conversational request unrelated to a company workflow.",
        }},
        "pairs": (
            ("Find firms offering automated underwriting to insurers.", "Why did our last review reject Acme?", "company_screening", "database_question"),
            ("Research Acme's latest product launch on the web.", "What did our saved research on Acme conclude?", "research_request", "database_question"),
            ("Create an Excel file of the shortlisted firms.", "Shortlist firms selling claims automation software.", "report_generation", "company_screening"),
            ("Compare Acme and Beacon for strategic fit.", "Write a report on the Acme and Beacon comparison we completed.", "company_comparison", "report_generation"),
            ("How was your weekend?", "Check current public sources for Beacon's customers.", "general_chat", "research_request"),
            ("Which vendors provide digital loan origination?", "Contrast Acme's and Beacon's loan products.", "company_screening", "company_comparison"),
            ("Show the revenue recorded for Acme in our database.", "Tell me a joke before we start.", "database_question", "general_chat"),
            ("Find recent news about Cedar's expansion.", "Generate a briefing from the saved Cedar assessment.", "research_request", "report_generation"),
        ),
    },
    "external_research": {
        "question": {"type": "noul", "instructions": "Must the system consult a current external source to answer this request accurately?"},
        "pairs": (
            ("Verify Acme's current customer list on its website.", "Read the customer list already saved in Acme's profile.", "yes", "no"),
            ("Has Beacon announced a funding round this week?", "What funding round is recorded in Beacon's saved profile?", "yes", "no"),
            ("Check whether Cedar still sells directly to consumers today.", "Does the attached Cedar evidence say it sells to consumers?", "yes", "no"),
            ("Find the latest regulatory filing for Delta.", "Summarize the filing already attached to Delta's record.", "yes", "no"),
            ("Check recent news for changes to Elm's product.", "What product was listed in our prior Elm assessment?", "yes", "no"),
            ("Confirm Fable's current geographic coverage online.", "Which geographies are listed in Fable's stored record?", "yes", "no"),
            ("Look up Gamma's present leadership on its site.", "Who is named in Gamma's saved management table?", "yes", "no"),
            ("Research whether Harbor was acquired yesterday.", "What acquisition status does the imported Harbor row contain?", "yes", "no"),
        ),
    },
    "expensive_reasoning": {
        "question": {"type": "noul", "instructions": "Does this request need substantial multi-step analysis from the stronger reasoning model?"},
        "pairs": (
            ("Compare three targets for product fit, synergies, and integration risk; defend the ranking.", "Show Acme's revenue and headquarters from the database.", "yes", "no"),
            ("Weigh end-market growth against margin pressure and propose an acquisition thesis.", "List firms with revenue above $50 million.", "yes", "no"),
            ("Reconcile conflicting evidence about Beacon and explain which sources should govern.", "Return Beacon's stored website URL.", "yes", "no"),
            ("Design a strategy for combining Cedar's platform with ours and identify execution risks.", "Export the current shortlist as CSV.", "yes", "no"),
            ("Assess whether Delta's product would cannibalize our portfolio and justify a decision.", "Count companies headquartered in Germany.", "yes", "no"),
            ("Explain why two analysts reached different investment conclusions on Elm.", "Read Elm's saved screening status.", "yes", "no"),
            ("Model likely acquisition synergies under three growth scenarios.", "Filter company rows to those founded after 2015.", "yes", "no"),
            ("Resolve contradictory customer evidence and recommend whether to advance Fable.", "Fetch Fable's sector label from the database.", "yes", "no"),
        ),
    },
    "search_type": {
        "question": {"type": "choice", "instructions": "Which information path should the orchestrator take first?", "criteria": {
            "structured_database": "Filter or read exact stored fields such as revenue, region, date, or status.",
            "semantic_retrieval": "Search descriptions or other free text by meaning.",
            "hybrid_semantic_structured": "Combine semantic company matching with exact structured filters.",
            "external_research": "Collect current facts from external sources.",
            "existing_evidence": "Use a saved assessment, cited passage, or prior analysis already in the workspace.",
            "general_reasoning": "Reason from the information already given without searching another source.",
        }},
        "pairs": (
            ("Find companies whose descriptions suggest fraud prevention for banks.", "Show all companies with revenue above $50 million.", "semantic_retrieval", "structured_database"),
            ("Find fraud prevention vendors with revenue above $50 million.", "Show all firms with revenue above $50 million.", "hybrid_semantic_structured", "structured_database"),
            ("What did our prior Acme review conclude?", "Find new public evidence about Acme's current customers.", "existing_evidence", "external_research"),
            ("Search descriptions for firms that automate underwriting.", "Open the saved underwriting evidence already cited for Acme.", "semantic_retrieval", "existing_evidence"),
            ("Explain which of these two supplied business models has more integration risk.", "Quote the integration-risk conclusion from our saved assessment.", "general_reasoning", "existing_evidence"),
            ("Find software providers for insurers founded after 2016.", "Find companies described as insurance software providers.", "hybrid_semantic_structured", "semantic_retrieval"),
            ("Check whether Cedar has expanded into Asia this month.", "Reason about the implications of the supplied Asia expansion plan.", "external_research", "general_reasoning"),
            ("List all UK companies with 50–200 employees.", "Find payment infrastructure companies in the UK with 50–200 employees.", "structured_database", "hybrid_semantic_structured"),
        ),
    },
    "evidence_sufficiency": {
        "question": {"type": "noul", "instructions": "Is the supplied internal evidence sufficient to answer the specific request without guessing?"},
        "pairs": (
            ("Acme's record names its product and insurer customers. Decide whether it sells to insurers.", "Acme's record names its product but gives no customer information. Decide whether it sells to insurers.", "yes", "no"),
            ("Beacon's saved profile gives revenue and fiscal year. Report that revenue.", "Beacon's profile has no revenue figure. Report its revenue.", "yes", "no"),
            ("Cedar's assessment includes a cited reason for rejection. Explain that reason.", "Cedar is marked rejected but has no reason or citation. Explain why.", "yes", "no"),
            ("Delta's filing states headquarters and operating countries. List them.", "Delta's row includes a website but no location evidence. List its operating countries.", "yes", "no"),
            ("Elm's product page saved yesterday describes its underwriting workflow. Classify the workflow.", "Elm's record contains only its name and industry code. Classify its underwriting workflow.", "yes", "no"),
            ("Fable's saved customer case study names three banks. Identify the customer segment.", "Fable's company description mentions software but no buyers. Identify its customer segment.", "yes", "no"),
            ("Gamma's latest stored assessment contains the compared target scores. State the higher score.", "Gamma's note says a comparison was made but omits the scores. State the higher score.", "yes", "no"),
            ("Harbor's cited source states it sells only to enterprises. Apply the consumer exclusion.", "Harbor's profile has no end-customer evidence. Apply the consumer exclusion.", "yes", "no"),
        ),
    },
    "direct_escalate": {
        "question": {"type": "choice", "instructions": "What should the orchestrator do next?", "criteria": {
            "direct_local_action": "Run a deterministic local operation or answer from clear saved evidence.",
            "use_stronger_model": "Escalate because the request requires substantial judgment or synthesis.",
            "clarification_required": "Ask for a missing company, period, criterion, or other essential detail.",
        }},
        "pairs": (
            ("Show Acme's saved revenue for 2024.", "Recommend whether to acquire Acme after weighing fit, risk, and synergies.", "direct_local_action", "use_stronger_model"),
            ("Export the named shortlist to Excel.", "Export the shortlist, but no shortlist or criteria have been specified.", "direct_local_action", "clarification_required"),
            ("Reconcile conflicting evidence and advise whether to reject Beacon.", "Compare it to the others, but no company or comparison set is named.", "use_stronger_model", "clarification_required"),
            ("Count UK companies in the stored table.", "Design a market entry thesis from these competing signals.", "direct_local_action", "use_stronger_model"),
            ("Which target is strategically strongest given these disputed assumptions?", "Find relevant companies, but the sector and market are missing.", "use_stronger_model", "clarification_required"),
            ("Read Cedar's saved screening reason.", "Tell me why they were rejected, but no company is identified.", "direct_local_action", "clarification_required"),
            ("List firms with recorded revenue above $50 million.", "Determine which target has the best long-term acquisition value.", "direct_local_action", "use_stronger_model"),
            ("Research the company, but its name is missing.", "Analyze the competitive effects and integration risks of the proposed deal.", "clarification_required", "use_stronger_model"),
        ),
    },
}


def build_rows(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Build 384 rows; every counterfactual pair and its variants share a split."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for task, spec in _FAMILIES.items():
        order = list(range(8))
        rng.shuffle(order)
        split_by_pair = {pair_index: ("dev" if rank < 4 else "test") for rank, pair_index in enumerate(order)}
        for pair_index, (left, right, left_gold, right_gold) in enumerate(spec["pairs"]):
            split = split_by_pair[pair_index]
            for variant_index, scenario in enumerate(_SCENARIOS):
                for side, (detail, gold) in enumerate(((left, left_gold), (right, right_gold))):
                    rows.append({
                        "id": f"{task}-t{pair_index:02d}-p{variant_index:02d}-c{side}",
                        "task": task,
                        "state": f"{scenario}\n{detail}",
                        "question": {"decision": dict(spec["question"])},
                        "gold": gold,
                        "split": split,
                        "tags": [task, f"template_{pair_index:02d}", f"pair_{variant_index:02d}", "counterfactual"],
                    })
    rng.shuffle(rows)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
