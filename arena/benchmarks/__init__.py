"""Benchmark datasets for RAG evaluation."""

from .base import Benchmark, BenchmarkSample
from .locomo import LoCoMoBenchmark
from .synthetic_multihop import SyntheticMultiHopBenchmark
from .natural_questions import NaturalQuestionsBenchmark
from .hotpotqa import HotpotQABenchmark
from .beir_subset import BEIRSubsetBenchmark
from .ms_marco import MSMarcoBenchmark

__all__ = [
    "Benchmark",
    "BenchmarkSample",
    "LoCoMoBenchmark",
    "SyntheticMultiHopBenchmark",
    "NaturalQuestionsBenchmark",
    "HotpotQABenchmark",
    "BEIRSubsetBenchmark",
    "MSMarcoBenchmark",
]
