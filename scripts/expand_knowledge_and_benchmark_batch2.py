"""
Batch 2 Dataset and Knowledge Expansion Script for HalluciGuard-OKF.
Expands dataset to over 80 claims with balanced classes and leakage-free coverage.
"""

import json
from pathlib import Path

curated_path = Path("knowledge/artifacts/curated_knowledge.json")
benchmark_path = Path("datasets/benchmark/benchmark_data.json")

with open(curated_path, "r", encoding="utf-8") as f:
    knowledge = json.load(f)

with open(benchmark_path, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

new_facts_batch2 = [
    {
        "identity": {"id": "OKF-SCI-008", "okf_version": "1.0"},
        "content": {
            "statement": "Humans typically have 46 chromosomes in 23 pairs in each diploid cell.",
            "entity": "Human chromosomes",
            "attribute": "chromosome_count",
            "value": 46,
            "category": "science"
        },
        "source": {
            "name": "National Human Genome Research Institute (NHGRI)",
            "url": "https://www.genome.gov/genetics-glossary/Chromosome",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Public Domain"
        },
        "provenance": {"chain": ["NHGRI Genome Glossary"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": None, "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "NHGRI"},
        "lifecycle": {"freshness_state": "CURRENT", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-HIST-005", "okf_version": "1.0"},
        "content": {
            "statement": "Yuri Gagarin became the first human in outer space on April 12, 1961 aboard Vostok 1.",
            "entity": "Yuri Gagarin",
            "attribute": "spaceflight_date",
            "value": "1961-04-12",
            "category": "history"
        },
        "source": {
            "name": "NASA History Division",
            "url": "https://history.nasa.gov",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Public Domain"
        },
        "provenance": {"chain": ["Soviet Space Agency Official Records"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1961-04-12T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "NASA History Office"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-HIST-006", "okf_version": "1.0"},
        "content": {
            "statement": "The RMS Titanic sank in the North Atlantic Ocean on April 15, 1912 after colliding with an iceberg.",
            "entity": "Titanic",
            "attribute": "sinking_date",
            "value": "1912-04-15",
            "category": "history"
        },
        "source": {
            "name": "British Board of Trade Inquiry",
            "url": "https://www.titanicinquiry.org",
            "source_type": "primary",
            "reliability_estimate": 0.99,
            "license": "Public Record"
        },
        "provenance": {"chain": ["British Board of Trade Inquiry 1912"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1912-04-15T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Board of Trade"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-TECH-006", "okf_version": "1.0"},
        "content": {
            "statement": "Google was founded on September 4, 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
            "entity": "Google",
            "attribute": "founding_date",
            "value": "1998-09-04",
            "category": "technology"
        },
        "source": {
            "name": "Alphabet Inc. Investor Relations & SEC Filings",
            "url": "https://abc.xyz/investor/",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Public Record"
        },
        "provenance": {"chain": ["SEC Form 10-K", "State of California Incorporation"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1998-09-04T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "SEC"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-TECH-007", "okf_version": "1.0"},
        "content": {
            "statement": "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen in Albuquerque, New Mexico.",
            "entity": "Microsoft",
            "attribute": "founding_date",
            "value": "1975-04-04",
            "category": "technology"
        },
        "source": {
            "name": "Microsoft Corporate History",
            "url": "https://www.microsoft.com/en-us/about",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Public Record"
        },
        "provenance": {"chain": ["State of New Mexico Incorporation Records 1975"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1975-04-04T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "State of New Mexico"},
        "lifecycle": {"freshness_state": "HISTORICAL", "ingested_at": "2024-01-01T00:00:00Z"}
    },
    {
        "identity": {"id": "OKF-GEO-005", "okf_version": "1.0"},
        "content": {
            "statement": "Tokyo is the official capital city of Japan, serving as the seat of the Japanese Emperor and government.",
            "entity": "Tokyo",
            "attribute": "capital_of_japan",
            "value": "Tokyo",
            "category": "geography"
        },
        "source": {
            "name": "Tokyo Metropolitan Government",
            "url": "https://www.metro.tokyo.lg.jp",
            "source_type": "primary",
            "reliability_estimate": 1.0,
            "license": "Official Government Record"
        },
        "provenance": {"chain": ["Constitution of Japan 1947"], "is_primary": True, "recorded_at": "2024-01-01T00:00:00Z"},
        "temporal": {"valid_from": "1868-10-23T00:00:00Z", "valid_until": None, "temporal_scope": "permanent"},
        "verification": {"status": "verified", "confidence": 1.0, "verifier": "Government of Japan"},
        "lifecycle": {"freshness_state": "CURRENT", "ingested_at": "2024-01-01T00:00:00Z"}
    }
]

existing_ids = {f["identity"]["id"] for f in knowledge}
for nf in new_facts_batch2:
    if nf["identity"]["id"] not in existing_ids:
        knowledge.append(nf)
        existing_ids.add(nf["identity"]["id"])

with open(curated_path, "w", encoding="utf-8") as f:
    json.dump(knowledge, f, indent=2)

new_scenarios_batch2 = [
    {
        "example_id": "BM-EXP-021",
        "domain": "science",
        "query": "How many chromosomes do humans have?",
        "llm_response": "Humans have 46 chromosomes in each somatic cell.",
        "claims": [
            {
                "claim_id": "C046-1",
                "claim_text": "Humans typically have 46 chromosomes in each diploid cell.",
                "claim_type": "numerical",
                "gold_label": "SUPPORTED",
                "evidence": "Humans typically have 46 chromosomes in 23 pairs in each diploid cell.",
                "evidence_source": "National Human Genome Research Institute (NHGRI)",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-022",
        "domain": "science",
        "query": "How many chromosomes do humans have?",
        "llm_response": "Humans have 48 chromosomes in each body cell.",
        "claims": [
            {
                "claim_id": "C047-1",
                "claim_text": "Humans typically have 48 chromosomes in each diploid cell.",
                "claim_type": "numerical",
                "gold_label": "CONTRADICTED",
                "evidence": "Humans typically have 46 chromosomes in 23 pairs in each diploid cell.",
                "evidence_source": "National Human Genome Research Institute (NHGRI)",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "easy",
                "hallucination_pattern": "wrong_number"
            }
        ]
    },
    {
        "example_id": "BM-EXP-023",
        "domain": "history",
        "query": "Who was the first human in space?",
        "llm_response": "Yuri Gagarin became the first human in outer space in 1961.",
        "claims": [
            {
                "claim_id": "C048-1",
                "claim_text": "Yuri Gagarin became the first human in outer space in 1961.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "Yuri Gagarin became the first human in outer space on April 12, 1961 aboard Vostok 1.",
                "evidence_source": "NASA History Division",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-024",
        "domain": "history",
        "query": "Who was the first human in space?",
        "llm_response": "John Glenn became the first human in outer space in 1959.",
        "claims": [
            {
                "claim_id": "C049-1",
                "claim_text": "John Glenn became the first human in outer space.",
                "claim_type": "attribution",
                "gold_label": "CONTRADICTED",
                "evidence": "Yuri Gagarin became the first human in outer space on April 12, 1961 aboard Vostok 1.",
                "evidence_source": "NASA History Division",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_attribution"
            },
            {
                "claim_id": "C049-2",
                "claim_text": "The first human spaceflight occurred in 1959.",
                "claim_type": "temporal",
                "gold_label": "CONTRADICTED",
                "evidence": "Yuri Gagarin became the first human in outer space on April 12, 1961 aboard Vostok 1.",
                "evidence_source": "NASA History Division",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_date"
            }
        ]
    },
    {
        "example_id": "BM-EXP-025",
        "domain": "history",
        "query": "When did Titanic sink?",
        "llm_response": "The Titanic sank on April 15, 1912 after striking an iceberg.",
        "claims": [
            {
                "claim_id": "C050-1",
                "claim_text": "The RMS Titanic sank on April 15, 1912.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "The RMS Titanic sank in the North Atlantic Ocean on April 15, 1912 after colliding with an iceberg.",
                "evidence_source": "British Board of Trade Inquiry",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-026",
        "domain": "history",
        "query": "When did Titanic sink?",
        "llm_response": "The Titanic sank in 1905 during a storm.",
        "claims": [
            {
                "claim_id": "C051-1",
                "claim_text": "The RMS Titanic sank in 1905.",
                "claim_type": "temporal",
                "gold_label": "CONTRADICTED",
                "evidence": "The RMS Titanic sank in the North Atlantic Ocean on April 15, 1912 after colliding with an iceberg.",
                "evidence_source": "British Board of Trade Inquiry",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "wrong_date"
            }
        ]
    },
    {
        "example_id": "BM-EXP-027",
        "domain": "technology",
        "query": "Who founded Google and when?",
        "llm_response": "Google was founded in 1998 by Larry Page and Sergey Brin.",
        "claims": [
            {
                "claim_id": "C052-1",
                "claim_text": "Google was founded in 1998.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "Google was founded on September 4, 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
                "evidence_source": "Alphabet Inc. Investor Relations & SEC Filings",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            },
            {
                "claim_id": "C052-2",
                "claim_text": "Google was founded by Larry Page and Sergey Brin.",
                "claim_type": "attribution",
                "gold_label": "SUPPORTED",
                "evidence": "Google was founded on September 4, 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
                "evidence_source": "Alphabet Inc. Investor Relations & SEC Filings",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-028",
        "domain": "technology",
        "query": "Who founded Google?",
        "llm_response": "Google was founded in 1991 by Mark Zuckerberg.",
        "claims": [
            {
                "claim_id": "C053-1",
                "claim_text": "Google was founded by Mark Zuckerberg.",
                "claim_type": "attribution",
                "gold_label": "CONTRADICTED",
                "evidence": "Google was founded on September 4, 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
                "evidence_source": "Alphabet Inc. Investor Relations & SEC Filings",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_attribution"
            },
            {
                "claim_id": "C053-2",
                "claim_text": "Google was founded in 1991.",
                "claim_type": "temporal",
                "gold_label": "CONTRADICTED",
                "evidence": "Google was founded on September 4, 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University.",
                "evidence_source": "Alphabet Inc. Investor Relations & SEC Filings",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_date"
            }
        ]
    },
    {
        "example_id": "BM-EXP-029",
        "domain": "technology",
        "query": "Who founded Microsoft and when?",
        "llm_response": "Microsoft was founded in 1975 by Bill Gates and Paul Allen.",
        "claims": [
            {
                "claim_id": "C054-1",
                "claim_text": "Microsoft was founded in 1975.",
                "claim_type": "temporal",
                "gold_label": "SUPPORTED",
                "evidence": "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen in Albuquerque, New Mexico.",
                "evidence_source": "Microsoft Corporate History",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            },
            {
                "claim_id": "C054-2",
                "claim_text": "Microsoft was founded by Bill Gates and Paul Allen.",
                "claim_type": "attribution",
                "gold_label": "SUPPORTED",
                "evidence": "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen in Albuquerque, New Mexico.",
                "evidence_source": "Microsoft Corporate History",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-030",
        "domain": "technology",
        "query": "Who founded Microsoft?",
        "llm_response": "Microsoft was founded in 1982 by Steve Jobs.",
        "claims": [
            {
                "claim_id": "C055-1",
                "claim_text": "Microsoft was founded by Steve Jobs.",
                "claim_type": "attribution",
                "gold_label": "CONTRADICTED",
                "evidence": "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen in Albuquerque, New Mexico.",
                "evidence_source": "Microsoft Corporate History",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_attribution"
            },
            {
                "claim_id": "C055-2",
                "claim_text": "Microsoft was founded in 1982.",
                "claim_type": "temporal",
                "gold_label": "CONTRADICTED",
                "evidence": "Microsoft was founded on April 4, 1975 by Bill Gates and Paul Allen in Albuquerque, New Mexico.",
                "evidence_source": "Microsoft Corporate History",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "HISTORICAL",
                "difficulty": "medium",
                "hallucination_pattern": "wrong_date"
            }
        ]
    },
    {
        "example_id": "BM-EXP-031",
        "domain": "geography",
        "query": "What is the capital of Japan?",
        "llm_response": "The capital city of Japan is Tokyo.",
        "claims": [
            {
                "claim_id": "C056-1",
                "claim_text": "The capital city of Japan is Tokyo.",
                "claim_type": "geographic",
                "gold_label": "SUPPORTED",
                "evidence": "Tokyo is the official capital city of Japan, serving as the seat of the Japanese Emperor and government.",
                "evidence_source": "Tokyo Metropolitan Government",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "easy",
                "hallucination_pattern": "faithful"
            }
        ]
    },
    {
        "example_id": "BM-EXP-032",
        "domain": "geography",
        "query": "What is the capital of Japan?",
        "llm_response": "The capital city of Japan is Kyoto.",
        "claims": [
            {
                "claim_id": "C057-1",
                "claim_text": "The capital city of Japan is Kyoto.",
                "claim_type": "geographic",
                "gold_label": "CONTRADICTED",
                "evidence": "Tokyo is the official capital city of Japan, serving as the seat of the Japanese Emperor and government.",
                "evidence_source": "Tokyo Metropolitan Government",
                "provenance": {"source_type": "primary", "verified": True, "timestamp": "2024-01-01"},
                "freshness": "CURRENT",
                "difficulty": "easy",
                "hallucination_pattern": "wrong_entity"
            }
        ]
    },
    {
        "example_id": "BM-EXP-033",
        "domain": "unverified",
        "query": "Who is the lead engineer of the Dyson swarm project?",
        "llm_response": "Dr. Helena Zhang has been the lead engineer of the Dyson swarm project at MIT since 2031.",
        "claims": [
            {
                "claim_id": "C058-1",
                "claim_text": "Dr. Helena Zhang has been the lead engineer of the Dyson swarm project at MIT since 2031.",
                "claim_type": "attribution",
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
        "example_id": "BM-EXP-034",
        "domain": "unverified",
        "query": "What is the cure for chronic chronosomnia?",
        "llm_response": "Chronic chronosomnia is cured using neuro-somnic frequency modulation developed by BioPhase in 2028.",
        "claims": [
            {
                "claim_id": "C059-1",
                "claim_text": "Chronic chronosomnia is cured using neuro-somnic frequency modulation developed by BioPhase in 2028.",
                "claim_type": "factual",
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
        "example_id": "BM-EXP-035",
        "domain": "unverified",
        "query": "Who directed the 2035 film Beyond the Horizon?",
        "llm_response": "Beyond the Horizon was directed by Christopher Nolan in 2035.",
        "claims": [
            {
                "claim_id": "C060-1",
                "claim_text": "Beyond the Horizon was directed by Christopher Nolan in 2035.",
                "claim_type": "temporal",
                "gold_label": "UNVERIFIED",
                "evidence": None,
                "evidence_source": None,
                "provenance": {},
                "freshness": "UNKNOWN",
                "difficulty": "easy",
                "hallucination_pattern": "future_fabrication"
            }
        ]
    }
]

existing_scen_ids = {s["example_id"] for s in benchmark}
for ns in new_scenarios_batch2:
    if ns["example_id"] not in existing_scen_ids:
        benchmark.append(ns)
        existing_scen_ids.add(ns["example_id"])

with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(benchmark, f, indent=2)

print(f"Final Curated Facts Count: {len(knowledge)}")
print(f"Final Benchmark Scenarios Count: {len(benchmark)}")
total_claims = sum(len(ex.get("claims", [])) for ex in benchmark)
print(f"Final Total Benchmark Claims: {total_claims}")
