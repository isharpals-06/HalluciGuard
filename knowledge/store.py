"""
OKF Knowledge Store.
Loads, validates, indexes, and queries OKF artifacts.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from okf.schemas.okf_spec import OKFArtifact
from okf.validator.validator import OKFValidator


class OKFKnowledgeStore:
    def __init__(self, artifacts_path: Optional[str] = None):
        self.artifacts_path = Path(artifacts_path) if artifacts_path else None
        self.validator = OKFValidator()
        self.artifacts: Dict[str, OKFArtifact] = {}
        self.entity_index: Dict[str, List[str]] = {}

        if self.artifacts_path and self.artifacts_path.exists():
            self.load_from_file(self.artifacts_path)

    def load_from_file(self, file_path: Path):
        """Loads and validates OKF artifacts from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        for item in data:
            is_valid, errors, artifact = self.validator.validate(item)
            if is_valid and artifact:
                self.add_artifact(artifact)
            else:
                print(f"[WARNING] Rejected invalid OKF artifact: {errors}")

    def add_artifact(self, artifact: OKFArtifact):
        """Stores a valid OKF artifact and updates entity indices."""
        self.artifacts[artifact.identity.id] = artifact
        ent = artifact.content.entity.lower()
        if ent not in self.entity_index:
            self.entity_index[ent] = []
        self.entity_index[ent].append(artifact.identity.id)

    def get_by_id(self, artifact_id: str) -> Optional[OKFArtifact]:
        return self.artifacts.get(artifact_id)

    def get_all(self) -> List[OKFArtifact]:
        return list(self.artifacts.values())

    def get_all_texts(self) -> List[str]:
        return [art.content.statement for art in self.artifacts.values()]

    def search_by_entity(self, entity_name: str) -> List[OKFArtifact]:
        matches = []
        query_ent = entity_name.lower().strip()
        for ent, ids in self.entity_index.items():
            if query_ent in ent or ent in query_ent:
                for art_id in ids:
                    matches.append(self.artifacts[art_id])
        return matches

    def count(self) -> int:
        return len(self.artifacts)
