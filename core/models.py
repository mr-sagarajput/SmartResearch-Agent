"""
Core Domain Data Models for SmartResearch Chatbot.
==================================================
Typed dataclasses with dictionary serialization and deserialization methods
for clean decoupling across core services, web UI, and CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResearchPillar:
    """Represents a single academic research pillar query, findings, and metadata."""

    pillar: str
    query: str = ""
    title: str = ""
    url: str = ""
    notes: str = ""
    full_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResearchPillar:
        return cls(
            pillar=str(data.get("pillar", "")),
            query=str(data.get("query", "")),
            title=str(data.get("title", "")),
            url=str(data.get("url", "")),
            notes=str(data.get("notes", "")),
            full_snippet=str(data.get("full_snippet", "")),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class SearchResult:
    """Represents a web search result or summarized article finding."""

    title: str
    url: str
    notes: str = ""
    full_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SearchResult:
        return cls(
            title=str(data.get("title", "")),
            url=str(data.get("url", "")),
            notes=str(data.get("notes", "")),
            full_snippet=str(data.get("full_snippet", "")),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class DocumentChunk:
    """Represents a text chunk parsed from a document page or synthesized web source."""

    chunk_id: int
    source: str
    page: int
    text: str
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentChunk:
        return cls(
            chunk_id=int(data.get("chunk_id", 0)),
            source=str(data.get("source", "")),
            page=int(data.get("page", 0)),
            text=str(data.get("text", "")),
            url=str(data.get("url", "")),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class VectorMatch:
    """Represents a retrieved document chunk paired with its cosine similarity score."""

    chunk: DocumentChunk
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict() if hasattr(self.chunk, "to_dict") else asdict(self.chunk),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VectorMatch:
        raw_chunk = data.get("chunk", {})
        if isinstance(raw_chunk, dict):
            chunk = DocumentChunk.from_dict(raw_chunk)
        elif isinstance(raw_chunk, DocumentChunk):
            chunk = raw_chunk
        else:
            chunk = DocumentChunk(chunk_id=0, source="unknown", page=0, text=str(raw_chunk))
        return cls(chunk=chunk, score=float(data.get("score", 0.0)))

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class RouterDecision:
    """Represents the outcome of autonomous query routing."""

    decision: str  # "VECTOR_HIT", "WEB_RESEARCH_TRIGGERED", "NOT_FOUND"
    answer: str
    sources: List[Any] = field(default_factory=list)
    max_score: float = 0.0
    is_web_research: bool = False
    provider: str = ""
    updated_embeddings: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "answer": self.answer,
            "sources": self.sources,
            "max_score": self.max_score,
            "is_web_research": self.is_web_research,
            "provider": self.provider,
            "updated_embeddings": self.updated_embeddings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RouterDecision:
        return cls(
            decision=str(data.get("decision", "")),
            answer=str(data.get("answer", "")),
            sources=list(data.get("sources", [])),
            max_score=float(data.get("max_score", 0.0)),
            is_web_research=bool(data.get("is_web_research", False)),
            provider=str(data.get("provider", "")),
            updated_embeddings=data.get("updated_embeddings"),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class DeepResearchResult:
    """Represents the result of 5-pillar deep academic research."""

    success: bool
    topic: str
    summary: str = ""
    pdf_path: Optional[str] = None
    sources: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    provider: str = ""
    updated_embeddings: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "topic": self.topic,
            "summary": self.summary,
            "pdf_path": self.pdf_path,
            "sources": self.sources,
            "error": self.error,
            "provider": self.provider,
            "updated_embeddings": self.updated_embeddings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeepResearchResult:
        return cls(
            success=bool(data.get("success", False)),
            topic=str(data.get("topic", "")),
            summary=str(data.get("summary", "")),
            pdf_path=data.get("pdf_path"),
            sources=list(data.get("sources", [])),
            error=data.get("error"),
            provider=str(data.get("provider", "")),
            updated_embeddings=data.get("updated_embeddings"),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ChatMessage:
    """Represents an individual message in a conversation session."""

    role: str
    content: str
    sources: Optional[List[Any]] = None
    provider: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.sources is not None:
            d["sources"] = self.sources
        if self.provider is not None:
            d["provider"] = self.provider
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatMessage:
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "")),
            sources=data.get("sources"),
            provider=data.get("provider"),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class ChatSessionState:
    """Represents the application conversational session state."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    research_messages: List[Dict[str, Any]] = field(default_factory=list)
    uploaded_pdf_name: Optional[str] = None
    file_hash: Optional[str] = None
    chunks_count: int = 0
    active_topic: str = ""
    last_pdf_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatSessionState:
        return cls(
            messages=list(data.get("messages", [])),
            research_messages=list(data.get("research_messages", [])),
            uploaded_pdf_name=data.get("uploaded_pdf_name"),
            file_hash=data.get("file_hash"),
            chunks_count=int(data.get("chunks_count", 0)),
            active_topic=str(data.get("active_topic", "")),
            last_pdf_path=data.get("last_pdf_path"),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
