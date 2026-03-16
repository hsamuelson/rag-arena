"""Configuration for the RAG Arena."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class OllamaConfig:
    """Ollama connection settings."""
    base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    chat_model: str = "qwen3.5:122b"
    embed_dimensions: int = 768
    timeout: int = 120


@dataclass
class ArenaConfig:
    """Top-level arena configuration."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    top_k: int = 10
    n_components: int = 3  # PCA components (SEMDA default)
    # Optional: when set, use a sentence-transformers model instead of Ollama
    # for dense embeddings.  Example values:
    #   "Snowflake/snowflake-arctic-embed-l"
    #   "Alibaba-NLP/gte-large-en-v1.5"
    st_embed_model: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "ArenaConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        ollama = OllamaConfig(**raw.get("ollama", {}))

        defaults = cls()
        return cls(
            ollama=ollama,
            data_dir=Path(raw.get("data_dir", str(defaults.data_dir))),
            results_dir=Path(raw.get("results_dir", str(defaults.results_dir))),
            top_k=raw.get("top_k", defaults.top_k),
            n_components=raw.get("n_components", defaults.n_components),
            st_embed_model=raw.get("st_embed_model", defaults.st_embed_model),
        )

    def to_yaml(self, path: Path) -> None:
        """Write config to a YAML file."""
        data = {
            "ollama": {
                "base_url": self.ollama.base_url,
                "embed_model": self.ollama.embed_model,
                "chat_model": self.ollama.chat_model,
                "embed_dimensions": self.ollama.embed_dimensions,
            },
            "data_dir": str(self.data_dir),
            "results_dir": str(self.results_dir),
            "top_k": self.top_k,
            "n_components": self.n_components,
        }
        if self.st_embed_model is not None:
            data["st_embed_model"] = self.st_embed_model
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
