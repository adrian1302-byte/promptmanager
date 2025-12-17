"""Prompt versioning system."""

import hashlib
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from pathlib import Path


class VersionStatus(Enum):
    """Status of a prompt version."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class PromptVersion:
    """A single version of a prompt."""
    version_id: str
    prompt_id: str
    content: str
    version_number: int
    status: VersionStatus = VersionStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = "system"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
        if not self.version_id:
            self.version_id = f"{self.prompt_id}_v{self.version_number}"

    def _compute_hash(self) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptVersion":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = VersionStatus(data.get("status", "draft"))
        return cls(**data)


@dataclass
class PromptHistory:
    """History of all versions for a prompt."""
    prompt_id: str
    name: str
    current_version: int = 0
    versions: List[PromptVersion] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_version(self, version_number: int) -> Optional[PromptVersion]:
        """Get a specific version."""
        for v in self.versions:
            if v.version_number == version_number:
                return v
        return None

    def get_active_version(self) -> Optional[PromptVersion]:
        """Get the currently active version."""
        for v in self.versions:
            if v.status == VersionStatus.ACTIVE:
                return v
        # If no active, return latest
        return self.versions[-1] if self.versions else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "name": self.name,
            "current_version": self.current_version,
            "versions": [v.to_dict() for v in self.versions],
            "tags": self.tags,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptHistory":
        """Create from dictionary."""
        versions = [PromptVersion.from_dict(v) for v in data.get("versions", [])]
        return cls(
            prompt_id=data["prompt_id"],
            name=data["name"],
            current_version=data.get("current_version", 0),
            versions=versions,
            tags=data.get("tags", []),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
        )


class VersioningBackend:
    """Abstract backend for version storage."""

    def save(self, history: PromptHistory) -> None:
        raise NotImplementedError

    def load(self, prompt_id: str) -> Optional[PromptHistory]:
        raise NotImplementedError

    def list_all(self) -> List[str]:
        raise NotImplementedError

    def delete(self, prompt_id: str) -> bool:
        raise NotImplementedError


class FileVersioningBackend(VersioningBackend):
    """File-based versioning backend."""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, prompt_id: str) -> Path:
        """Get file path for a prompt."""
        safe_id = prompt_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_id}.json"

    def save(self, history: PromptHistory) -> None:
        """Save prompt history to file."""
        path = self._get_path(history.prompt_id)
        with open(path, "w") as f:
            json.dump(history.to_dict(), f, indent=2)

    def load(self, prompt_id: str) -> Optional[PromptHistory]:
        """Load prompt history from file."""
        path = self._get_path(prompt_id)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return PromptHistory.from_dict(data)

    def list_all(self) -> List[str]:
        """List all prompt IDs."""
        return [
            p.stem for p in self.storage_dir.glob("*.json")
        ]

    def delete(self, prompt_id: str) -> bool:
        """Delete a prompt and its history."""
        path = self._get_path(prompt_id)
        if path.exists():
            path.unlink()
            return True
        return False


class MemoryVersioningBackend(VersioningBackend):
    """In-memory versioning backend for testing."""

    def __init__(self):
        self._storage: Dict[str, PromptHistory] = {}

    def save(self, history: PromptHistory) -> None:
        self._storage[history.prompt_id] = history

    def load(self, prompt_id: str) -> Optional[PromptHistory]:
        return self._storage.get(prompt_id)

    def list_all(self) -> List[str]:
        return list(self._storage.keys())

    def delete(self, prompt_id: str) -> bool:
        if prompt_id in self._storage:
            del self._storage[prompt_id]
            return True
        return False


class PromptVersioning:
    """
    Prompt versioning manager.

    Handles version creation, retrieval, and management.
    """

    def __init__(self, backend: Optional[VersioningBackend] = None):
        """
        Initialize versioning manager.

        Args:
            backend: Storage backend (defaults to in-memory)
        """
        self.backend = backend or MemoryVersioningBackend()

    def create_prompt(
        self,
        prompt_id: str,
        name: str,
        content: str,
        description: str = "",
        created_by: str = "system",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptVersion:
        """
        Create a new prompt with initial version.

        Args:
            prompt_id: Unique prompt identifier
            name: Human-readable name
            content: Prompt content
            description: Version description
            created_by: Author identifier
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            The created PromptVersion
        """
        # Check if prompt already exists
        existing = self.backend.load(prompt_id)
        if existing:
            raise ValueError(f"Prompt '{prompt_id}' already exists")

        # Create first version
        version = PromptVersion(
            version_id=f"{prompt_id}_v1",
            prompt_id=prompt_id,
            content=content,
            version_number=1,
            status=VersionStatus.ACTIVE,
            description=description,
            created_by=created_by,
            metadata=metadata or {}
        )

        # Create history
        history = PromptHistory(
            prompt_id=prompt_id,
            name=name,
            current_version=1,
            versions=[version],
            tags=tags or [],
            metadata=metadata or {}
        )

        self.backend.save(history)
        return version

    def create_version(
        self,
        prompt_id: str,
        content: str,
        description: str = "",
        created_by: str = "system",
        auto_activate: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptVersion:
        """
        Create a new version of an existing prompt.

        Args:
            prompt_id: Prompt identifier
            content: New content
            description: Version description
            created_by: Author identifier
            auto_activate: Whether to automatically activate this version
            metadata: Optional metadata

        Returns:
            The created PromptVersion
        """
        history = self.backend.load(prompt_id)
        if not history:
            raise ValueError(f"Prompt '{prompt_id}' not found")

        # Check for duplicate content
        latest = history.versions[-1] if history.versions else None
        new_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        if latest and latest.content_hash == new_hash:
            raise ValueError("Content unchanged from latest version")

        # Create new version
        new_version_number = history.current_version + 1
        version = PromptVersion(
            version_id=f"{prompt_id}_v{new_version_number}",
            prompt_id=prompt_id,
            content=content,
            version_number=new_version_number,
            status=VersionStatus.ACTIVE if auto_activate else VersionStatus.DRAFT,
            description=description,
            created_by=created_by,
            parent_version=latest.version_id if latest else None,
            metadata=metadata or {}
        )

        # If auto-activating, deactivate previous active version
        if auto_activate:
            for v in history.versions:
                if v.status == VersionStatus.ACTIVE:
                    v.status = VersionStatus.DEPRECATED

        history.versions.append(version)
        history.current_version = new_version_number

        self.backend.save(history)
        return version

    def activate_version(self, prompt_id: str, version_number: int) -> PromptVersion:
        """
        Activate a specific version.

        Args:
            prompt_id: Prompt identifier
            version_number: Version to activate

        Returns:
            The activated version
        """
        history = self.backend.load(prompt_id)
        if not history:
            raise ValueError(f"Prompt '{prompt_id}' not found")

        target = None
        for v in history.versions:
            if v.version_number == version_number:
                target = v
            elif v.status == VersionStatus.ACTIVE:
                v.status = VersionStatus.DEPRECATED

        if not target:
            raise ValueError(f"Version {version_number} not found")

        target.status = VersionStatus.ACTIVE
        self.backend.save(history)
        return target

    def get_version(
        self,
        prompt_id: str,
        version_number: Optional[int] = None
    ) -> Optional[PromptVersion]:
        """
        Get a specific version or the active version.

        Args:
            prompt_id: Prompt identifier
            version_number: Specific version (None for active)

        Returns:
            The PromptVersion or None
        """
        history = self.backend.load(prompt_id)
        if not history:
            return None

        if version_number is not None:
            return history.get_version(version_number)
        return history.get_active_version()

    def get_content(
        self,
        prompt_id: str,
        version_number: Optional[int] = None
    ) -> Optional[str]:
        """Get prompt content for a version."""
        version = self.get_version(prompt_id, version_number)
        return version.content if version else None

    def get_history(self, prompt_id: str) -> Optional[PromptHistory]:
        """Get full version history for a prompt."""
        return self.backend.load(prompt_id)

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all prompts with summary info."""
        result = []
        for prompt_id in self.backend.list_all():
            history = self.backend.load(prompt_id)
            if history:
                active = history.get_active_version()
                result.append({
                    "prompt_id": history.prompt_id,
                    "name": history.name,
                    "current_version": history.current_version,
                    "active_version": active.version_number if active else None,
                    "total_versions": len(history.versions),
                    "tags": history.tags,
                    "created_at": history.created_at,
                })
        return result

    def compare_versions(
        self,
        prompt_id: str,
        version_a: int,
        version_b: int
    ) -> Dict[str, Any]:
        """
        Compare two versions of a prompt.

        Args:
            prompt_id: Prompt identifier
            version_a: First version number
            version_b: Second version number

        Returns:
            Comparison result with diff info
        """
        history = self.backend.load(prompt_id)
        if not history:
            raise ValueError(f"Prompt '{prompt_id}' not found")

        va = history.get_version(version_a)
        vb = history.get_version(version_b)

        if not va or not vb:
            raise ValueError("One or both versions not found")

        return {
            "prompt_id": prompt_id,
            "version_a": {
                "number": va.version_number,
                "content": va.content,
                "created_at": va.created_at,
                "status": va.status.value,
            },
            "version_b": {
                "number": vb.version_number,
                "content": vb.content,
                "created_at": vb.created_at,
                "status": vb.status.value,
            },
            "content_changed": va.content_hash != vb.content_hash,
            "length_diff": len(vb.content) - len(va.content),
        }

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt and all its versions."""
        return self.backend.delete(prompt_id)

    def tag_prompt(self, prompt_id: str, tags: List[str]) -> None:
        """Add tags to a prompt."""
        history = self.backend.load(prompt_id)
        if not history:
            raise ValueError(f"Prompt '{prompt_id}' not found")

        history.tags = list(set(history.tags + tags))
        self.backend.save(history)

    def search_by_tag(self, tag: str) -> List[str]:
        """Find prompts with a specific tag."""
        result = []
        for prompt_id in self.backend.list_all():
            history = self.backend.load(prompt_id)
            if history and tag in history.tags:
                result.append(prompt_id)
        return result
