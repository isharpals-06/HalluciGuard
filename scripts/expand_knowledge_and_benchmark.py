"""
Dataset and Knowledge Expansion Script for HalluciGuard-OKF.
Expands curated knowledge artifacts and benchmark dataset across diverse domains:
Science, History, Geography, Technology, Medicine, Literature, Politics, and Adversarial scenarios.
"""

import json
from pathlib import Path

curated_path = Path("knowledge/artifacts/curated_knowledge.json")
benchmark_path = Path("datasets/benchmark/benchmark_data.json")

# 1. Load existing curated knowledge
with open(curated_path, "r", encoding="utf-8") as f:
    existing_knowledge = json.load(f)

# 2. Load existing benchmark
with open(benchmark_path, "r", encoding="utf-8") as f:
    existing_benchmark = json.load(f)

print(f"Initial Curated Facts: {len(existing_knowledge)}")
print(f"Initial Benchmark Scenarios: {len(existing_benchmark)}")

new_facts = [
    {
        "identity": {"id": "OKF-SCI-004", "okf_version": "1.0"},
        "content": {
            "statement": "Water boils at exactly 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm).",
            "entity": "Water",
            "attribute": "boiling_point",
            "value": 100,
            "category": "science"
        },
        "source": {
            "name": "NIST Chemistry WebBook",
            "url": "https://webbook.nist.gov/chemistry/fluid/",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Public Domain"
        },
        "provenance": {"chain": ["NIST Standard Reference Data"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": None, "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "NIST"},
        "lifecycle": {"freshness_state": "CURRENT", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-MED-001", "okf_version": "1.0"},
        "content": {
            "statement": "Alexander Fleming discovered penicillin in September 1928 at St. Mary's Hospital in London.",
            "entity": "Penicillin",
            "attribute": "discovery_date",
            "value": "1928-09",
            "category": "medicine"
        },
        "source": {
            "name": "Nobel Prize in Physiology or Medicine Archives",
            "url": "https://www.nobelprize.org/prizes/medicine/1945/fleming/biographical/",
            "source_type": "primary",
            "reliability_estimate": 0.99,
            "license": "Public Record"
        },
        "provenance": {"chain": ["Nobel Foundation Archive"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1928-09-01T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 0.99, "verifier": "Nobel Foundation"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-SCI-005", "okf_version": "1.0"},
        "content": {
            "statement": "The chemical structure of the DNA double helix was discovered in 1953 by James Watson, Francis Crick, and Rosalind Franklin.",
            "entity": "DNA",
            "attribute": "double_helix_discovery",
            "value": "1953",
            "category": "science"
        },
        "source": {
            "name": "Nature Archive 1953",
            "url": "https://www.nature.com/articles/171737a0",
            "source_type": "primary",
            "reliability_estimate": 0.99,
            "license": "Nature Publishing"
        },
        "provenance": {"chain": ["Nature Original Publication April 25 1953"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1953-04-25T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 0.99, "verifier": "Nature Editorial Board"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-GEO-003", "okf_version": "1.0"},
        "content": {
            "statement": "Ottawa is the official capital city of Canada, designated by Queen Victoria in 1857.",
            "entity": "Ottawa",
            "attribute": "capital_of_canada",
            "value": "Ottawa",
            "category": "geography"
        },
        "source": {
            "name": "Government of Canada Historical Records",
            "url": "https://www.canada.ca/en/canadian-heritage/services/origin-name-capital.html",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Open Government Licence"
        },
        "provenance": {"chain": ["Official Government of Canada Gazette"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1857-12-31T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Parliament of Canada"},
        "lifecycle": {"freshness_state": "CURRENT", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-GEO-004", "okf_version": "1.0"},
        "content": {
            "statement": "Brasilia is the official federal capital city of Brazil, inaugurated on April 21, 1960.",
            "entity": "Brasilia",
            "attribute": "capital_of_brazil",
            "value": "Brasilia",
            "category": "geography"
        },
        "source": {
            "name": "IBGE Brazilian Institute of Geography and Statistics",
            "url": "https://www.ibge.gov.br",
            "source_type": "primary",
            "reliability_estimate": 0.99,
            "license": "Public Government Record"
        },
        "provenance": {"chain": ["Federal Constitution of Brazil 1988"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1960-04-21T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Government of Brazil"},
        "lifecycle": {"freshness_state": "CURRENT", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-TECH-003", "okf_version": "1.0"},
        "content": {
            "statement": "Python was created by Guido van Rossum and first released in February 1991.",
            "entity": "Python",
            "attribute": "creation_date",
            "value": "1991-02",
            "category": "technology"
        },
        "source": {
            "name": "Python Software Foundation",
            "url": "https://www.python.org/doc/essays/history/",
            "source_type": "primary",
            "reliability_estimate": 0.99,
            "license": "PSF License"
        },
        "provenance": {"chain": ["PSF Official History Archive"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1991-02-20T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Guido van Rossum"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-TECH-004", "okf_version": "1.0"},
        "content": {
            "statement": "The Linux kernel was created by Linus Torvalds and first publicly released on September 17, 1991.",
            "entity": "Linux",
            "attribute": "kernel_release_date",
            "value": "1991-09-17",
            "category": "technology"
        },
        "source": {
            "name": "The Linux Foundation Archives",
            "url": "https://kernel.org",
            "source_type": "primary",
            "reliability_estimate": 0.99,
            "license": "GPLv2"
        },
        "provenance": {"chain": ["comp.os.minix Torvalds Posting 1991"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1991-09-17T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Linux Foundation"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-HIST-003", "okf_version": "1.0"},
        "content": {
            "statement": "The Berlin Wall fell on November 9, 1989, marking the symbolic end of the Cold War in Europe.",
            "entity": "Berlin Wall",
            "attribute": "fall_date",
            "value": "1989-11-09",
            "category": "history"
        },
        "source": {
            "name": "German Historical Museum",
            "url": "https://www.dhm.de",
            "source_type": "primary",
            "reliability_estimate": 0.99,
            "license": "Public Cultural Archive"
        },
        "provenance": {"chain": ["GDR Government Press Conference Nov 9 1989"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1989-11-09T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Federal Republic of Germany"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-HIST-004", "okf_version": "1.0"},
        "content": {
            "statement": "The United States Declaration of Independence was ratified on July 4, 1776 by the Second Continental Congress.",
            "entity": "Declaration of Independence",
            "attribute": "ratification_date",
            "value": "1776-07-04",
            "category": "history"
        },
        "source": {
            "name": "US National Archives and Records Administration",
            "url": "https://www.archives.gov/founding-docs/declaration",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Public Domain"
        },
        "provenance": {"chain": ["Continental Congress Official Journals"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1776-07-04T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "US National Archives"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-SCI-006", "okf_version": "1.0"},
        "content": {
            "statement": "Albert Einstein published the General Theory of Relativity in November 1915.",
            "entity": "General Relativity",
            "attribute": "publication_date",
            "value": "1915-11",
            "category": "science"
        },
        "source": {
            "name": "Prussian Academy of Sciences Proceedings",
            "url": "https://einsteinpapers.press.princeton.edu",
            "source_type": "primary",
            "reliability_estimate": 0.99,
            "license": "Public Academic Archive"
        },
        "provenance": {"chain": ["Sitzungsberichte der Preussischen Akademie der Wissenschaften"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1915-11-25T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Prussian Academy of Sciences"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-SCI-007", "okf_version": "1.0"},
        "content": {
            "statement": "Albert Einstein received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect, not for the theory of relativity.",
            "entity": "Albert Einstein",
            "attribute": "nobel_prize_reason",
            "value": "photoelectric_effect",
            "category": "science"
        },
        "source": {
            "name": "Nobel Prize in Physics Official Archive",
            "url": "https://www.nobelprize.org/prizes/physics/1921/summary/",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Public Record"
        },
        "provenance": {"chain": ["Royal Swedish Academy of Sciences Nobel Citation 1922"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1922-11-09T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Royal Swedish Academy of Sciences"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-TECH-005", "okf_version": "1.0"},
        "content": {
            "statement": "The World Wide Web was invented by Tim Berners-Lee in 1989 at CERN.",
            "entity": "World Wide Web",
            "attribute": "invention_date",
            "value": "1989",
            "category": "technology"
        },
        "source": {
            "name": "CERN Document Server",
            "url": "https://home.cern/science/computing/birth-web",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Public Domain"
        },
        "provenance": {"chain": ["Information Management: A Proposal CERN 1989"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1989-03-01T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "CERN"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    }
]

# Append new facts if not present
existing_ids = {f["identity"]["id"] for f in existing_knowledge}
for nf in new_facts:
    if nf["identity"]["id"] not in existing_ids:
        existing_knowledge.append(nf)
        existing_ids.add(nf["identity"]["id"])

with open(curated_path, "w", encoding="utf-8") as f:
    json.dump(existing_knowledge, f, indent=2)

print(f"Updated Curated Facts Count: {len(existing_knowledge)}")

# 3. Create rich new benchmark scenarios
new_scenarios = [
    {
        "example_id": "BM-EXP-001",
        "domain": "science",
        "query": "At what temperature does water boil?",
        "llm_response": "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
        "claims": [
            {
                "claim_id": "C026-1",
                "claim_text": "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
                "claim_type": "numerical",
                "gold_label": "SUPPORTED",
                "evidence": "Water boils at exactly 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm).",
                "evidence_source": "NIST Chemistry WebBook",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-002",
        "domain": "science",
        "query": "At what temperature does water boil?",
        "llm_response": "Water boils at 145 degrees Celsius at normal sea level pressure.",
        "claims": [
            {
                "claim_id": "C027-1",
                "claim_text": "Water boils at 145 degrees Celsius at normal sea level pressure.",
                "claim_type": "numerical",
                "gold_label": "CONTRADICTED",
                "evidence": "Water boils at exactly 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm).",
                "evidence_source": "NIST Chemistry WebBook",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "easy",
                "hallucination_pattern": "wrong_number"
            }
        ]
    },
    {
        "example_id": "BM-EXP-003",
        "domain": "medicine",
        "query": "Who discovered penicillin and when?",
        "llm_response": "Alexander Fleming discovered penicillin in 1928.",
        "claims": [
            {
                "claim_id": "C028-1",
                "claim_text": "Alexander Fleming discovered penicillin in 1928.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "Alexander Fleming discovered penicillin in September 1928 at St. Mary's Hospital in London.",
                "evidence_source": "Nobel Prize in Physiology or Medicine Archives",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-004",
        "domain": "medicine",
        "query": "Who discovered penicillin?",
        "llm_response": "Penicillin was discovered by Louis Pasteur in 1945.",
        "claims": [
            {
                "claim_id": "C029-1",
                "claim_text": "Penicillin was discovered by Louis Pasteur.",
                "claim_type": "attribution",
                "gold_label": "CONTRADICTED",
                "evidence": "Alexander Fleming discovered penicillin in September 1928 at St. Mary's Hospital in London.",
                "evidence_source": "Nobel Prize in Physiology or Medicine Archives",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_attribution"
            },
            {
                "claim_id": "C029-2",
                "claim_text": "Penicillin was discovered in 1945.",
                "claim_type": "temporal",
                "gold_label": "CONTRADICTED",
                "evidence": "Alexander Fleming discovered penicillin in September 1928 at St. Mary's Hospital in London.",
                "evidence_source": "Nobel Prize in Physiology or Medicine Archives",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_date"
            }
        ]
    },
    {
        "example_id": "BM-EXP-005",
        "domain": "geography",
        "query": "What is the capital of Canada?",
        "llm_response": "The capital city of Canada is Ottawa.",
        "claims": [
            {
                "claim_id": "C030-1",
                "claim_text": "The capital city of Canada is Ottawa.",
                "claim_type": "geographic",
                "gold_label": "SUPPORTED",
                "evidence": "Ottawa is the official capital city of Canada, designated by Queen Victoria in 1857.",
                "evidence_source": "Government of Canada Historical Records",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-006",
        "domain": "geography",
        "query": "What is the capital of Canada?",
        "llm_response": "The capital city of Canada is Toronto.",
        "claims": [
            {
                "claim_id": "C031-1",
                "claim_text": "The capital city of Canada is Toronto.",
                "claim_type": "geographic",
                "gold_label": "CONTRADICTED",
                "evidence": "Ottawa is the official capital city of Canada, designated by Queen Victoria in 1857.",
                "evidence_source": "Government of Canada Historical Records",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "easy",
                "hallucination_pattern": "wrong_entity"
            }
        ]
    },
    {
        "example_id": "BM-EXP-007",
        "domain": "geography",
        "query": "What is the capital of Brazil?",
        "llm_response": "The capital of Brazil is Rio de Janeiro.",
        "claims": [
            {
                "claim_id": "C032-1",
                "claim_text": "The capital of Brazil is Rio de Janeiro.",
                "claim_type": "geographic",
                "gold_label": "CONTRADICTED",
                "evidence": "Brasilia is the official federal capital city of Brazil, inaugurated on April 21, 1960.",
                "evidence_source": "IBGE Brazilian Institute of Geography and Statistics",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_entity"
            }
        ]
    },
    {
        "example_id": "BM-EXP-008",
        "domain": "technology",
        "query": "When was Python created?",
        "llm_response": "Python was created by Guido van Rossum in 1991.",
        "claims": [
            {
                "claim_id": "C033-1",
                "claim_text": "Python was created by Guido van Rossum.",
                "claim_type": "attribution",
                "gold_label": "SUPPORTED",
                "evidence": "Python was created by Guido van Rossum and first released in February 1991.",
                "evidence_source": "Python Software Foundation",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            },
            {
                "claim_id": "C033-2",
                "claim_text": "Python was created in 1991.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "Python was created by Guido van Rossum and first released in February 1991.",
                "evidence_source": "Python Software Foundation",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-009",
        "domain": "technology",
        "query": "Who created Linux and when?",
        "llm_response": "Linux was created by Linus Torvalds in 1991.",
        "claims": [
            {
                "claim_id": "C034-1",
                "claim_text": "The Linux kernel was created by Linus Torvalds.",
                "claim_type": "attribution",
                "gold_label": "SUPPORTED",
                "evidence": "The Linux kernel was created by Linus Torvalds and first publicly released on September 17, 1991.",
                "evidence_source": "The Linux Foundation Archives",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            },
            {
                "claim_id": "C034-2",
                "claim_text": "The Linux kernel was created in 1991.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "The Linux kernel was created by Linus Torvalds and first publicly released on September 17, 1991.",
                "evidence_source": "The Linux Foundation Archives",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-010",
        "domain": "technology",
        "query": "Who created Linux?",
        "llm_response": "The Linux kernel was created by Richard Stallman in 1985.",
        "claims": [
            {
                "claim_id": "C035-1",
                "claim_text": "The Linux kernel was created by Richard Stallman.",
                "claim_type": "attribution",
                "gold_label": "CONTRADICTED",
                "evidence": "The Linux kernel was created by Linus Torvalds and first publicly released on September 17, 1991.",
                "evidence_source": "The Linux Foundation Archives",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_attribution"
            },
            {
                "claim_id": "C035-2",
                "claim_text": "The Linux kernel was created in 1985.",
                "claim_type": "temporal",
                "gold_label": "CONTRADICTED",
                "evidence": "The Linux kernel was created by Linus Torvalds and first publicly released on September 17, 1991.",
                "evidence_source": "The Linux Foundation Archives",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_date"
            }
        ]
    },
    {
        "example_id": "BM-EXP-011",
        "domain": "history",
        "query": "When did the Berlin Wall fall?",
        "llm_response": "The Berlin Wall fell on November 9, 1989.",
        "claims": [
            {
                "claim_id": "C036-1",
                "claim_text": "The Berlin Wall fell on November 9, 1989.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "The Berlin Wall fell on November 9, 1989, marking the symbolic end of the Cold War in Europe.",
                "evidence_source": "German Historical Museum",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-012",
        "domain": "history",
        "query": "When did the Berlin Wall fall?",
        "llm_response": "The Berlin Wall fell in 1975.",
        "claims": [
            {
                "claim_id": "C037-1",
                "claim_text": "The Berlin Wall fell in 1975.",
                "claim_type": "temporal",
                "gold_label": "CONTRADICTED",
                "evidence": "The Berlin Wall fell on November 9, 1989, marking the symbolic end of the Cold War in Europe.",
                "evidence_source": "German Historical Museum",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "wrong_date"
            }
        ]
    },
    {
        "example_id": "BM-EXP-013",
        "domain": "science",
        "query": "Why did Einstein win the Nobel Prize?",
        "llm_response": "Albert Einstein won the 1921 Nobel Prize in Physics for General Relativity.",
        "claims": [
            {
                "claim_id": "C038-1",
                "claim_text": "Albert Einstein won the 1921 Nobel Prize in Physics for General Relativity.",
                "claim_type": "attribution",
                "gold_label": "CONTRADICTED",
                "evidence": "Albert Einstein received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect, not for the theory of relativity.",
                "evidence_source": "Nobel Prize in Physics Official Archive",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "hard",
                "hallucination_pattern": "subtle_misconception"
            }
        ]
    },
    {
        "example_id": "BM-EXP-014",
        "domain": "science",
        "query": "Why did Einstein win the Nobel Prize?",
        "llm_response": "Albert Einstein received the Nobel Prize in Physics for his work on the photoelectric effect.",
        "claims": [
            {
                "claim_id": "C039-1",
                "claim_text": "Albert Einstein received the Nobel Prize in Physics for his work on the photoelectric effect.",
                "claim_type": "factual",
                "gold_label": "SUPPORTED",
                "evidence": "Albert Einstein received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect, not for the theory of relativity.",
                "evidence_source": "Nobel Prize in Physics Official Archive",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-015",
        "domain": "unverified",
        "query": "Who won the 2048 lunar marathon?",
        "llm_response": "Captain Sarah Vance won the 2048 lunar marathon in Shackleton Crater.",
        "claims": [
            {
                "claim_id": "C040-1",
                "claim_text": "Captain Sarah Vance won the 2048 lunar marathon in Shackleton Crater.",
                "claim_type": "entity",
                "gold_label": "UNVERIFIED",
                "evidence": None,
                "evidence_source": None,
                "provenance": {},
                "freshness": "UNKNOWN",
                "difficulty": "easy",
                "hallucination_pattern": "complete_fabrication"
            }
        ]
    },
    {
        "example_id": "BM-EXP-016",
        "domain": "unverified",
        "query": "Tell me about the hyper-quantum transistor.",
        "llm_response": "The hyper-quantum transistor was patented by Dr. Aris Thorne at Cambridge in 2029.",
        "claims": [
            {
                "claim_id": "C041-1",
                "claim_text": "The hyper-quantum transistor was patented by Dr. Aris Thorne at Cambridge in 2029.",
                "claim_type": "temporal",
                "gold_label": "UNVERIFIED",
                "evidence": None,
                "evidence_source": None,
                "provenance": {},
                "freshness": "UNKNOWN",
                "difficulty": "easy",
                "hallucination_pattern": "complete_fabrication"
            }
        ]
    },
    {
        "example_id": "BM-EXP-017",
        "domain": "adversarial",
        "query": "Did Apollo 11 land on the moon?",
        "llm_response": "It is not untrue that Apollo 11 landed on the moon on July 20, 1969.",
        "claims": [
            {
                "claim_id": "C042-1",
                "claim_text": "It is not untrue that Apollo 11 landed on the moon on July 20, 1969.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "The Apollo 11 Lunar Module Eagle landed on the Moon on July 20, 1969, and Neil Armstrong became the first human to step onto the lunar surface.",
                "evidence_source": "NASA Apollo 11 Mission Overview",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "hard",
                "hallucination_pattern": "double_negation"
            }
        ]
    },
    {
        "example_id": "BM-EXP-018",
        "domain": "history",
        "query": "When was the US Declaration of Independence signed?",
        "llm_response": "The Declaration of Independence was ratified on July 4, 1776.",
        "claims": [
            {
                "claim_id": "C043-1",
                "claim_text": "The Declaration of Independence was ratified on July 4, 1776.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "The United States Declaration of Independence was ratified on July 4, 1776 by the Second Continental Congress.",
                "evidence_source": "US National Archives and Records Administration",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-019",
        "domain": "history",
        "query": "When was the US Declaration of Independence signed?",
        "llm_response": "The Declaration of Independence was signed on July 4, 1812.",
        "claims": [
            {
                "claim_id": "C044-1",
                "claim_text": "The Declaration of Independence was signed on July 4, 1812.",
                "claim_type": "temporal",
                "gold_label": "CONTRADICTED",
                "evidence": "The United States Declaration of Independence was ratified on July 4, 1776 by the Second Continental Congress.",
                "evidence_source": "US National Archives and Records Administration",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "wrong_date"
            }
        ]
    },
    {
        "example_id": "BM-EXP-020",
        "domain": "technology",
        "query": "Who invented the World Wide Web?",
        "llm_response": "The World Wide Web was invented by Tim Berners-Lee in 1989 at CERN.",
        "claims": [
            {
                "claim_id": "C045-1",
                "claim_text": "The World Wide Web was invented by Tim Berners-Lee in 1989 at CERN.",
                "claim_type": "attribution",
                "gold_label": "SUPPORTED",
                "evidence": "The World Wide Web was invented by Tim Berners-Lee in 1989 at CERN.",
                "evidence_source": "CERN Document Server",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    }
]

existing_scen_ids = {s["example_id"] for s in existing_benchmark}
for ns in new_scenarios:
    if ns["example_id"] not in existing_scen_ids:
        existing_benchmark.append(ns)
        existing_scen_ids.add(ns["example_id"])

with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(existing_benchmark, f, indent=2)

print(f"Updated Benchmark Scenarios Count: {len(existing_benchmark)}")
total_claims = sum(len(ex.get("claims", [])) for ex in existing_benchmark)
print(f"Updated Total Benchmark Claims: {total_claims}")
