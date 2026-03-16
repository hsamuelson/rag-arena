"""Synthetic multi-hop benchmark — generates controllable multi-hop retrieval tasks.

Unlike LoCoMo (which is a fixed dataset), this benchmark generates
synthetic documents and questions where we control:
  - Number of hops required
  - Corpus size (to test scaling)
  - Distractor density
  - Topic overlap

This lets us isolate specific retrieval weaknesses (e.g., "does this
hypothesis handle 10K files without degradation?").
"""

import hashlib
import random
from dataclasses import dataclass

from .base import Benchmark, BenchmarkSample


@dataclass
class SyntheticConfig:
    """Controls synthetic benchmark generation."""
    num_chains: int = 20           # Number of reasoning chains
    hops_per_chain: int = 3        # Documents per chain
    distractors_per_chain: int = 5 # Irrelevant docs per chain
    seed: int = 42
    topic_overlap: float = 0.3     # Fraction of shared vocabulary between chains


# Pre-defined topics and facts for generating realistic-ish documents
_TOPICS = [
    ("quantum computing", ["qubit", "superposition", "entanglement", "decoherence", "gate"]),
    ("climate science", ["carbon", "temperature", "glacier", "emission", "albedo"]),
    ("neuroscience", ["neuron", "synapse", "cortex", "dopamine", "plasticity"]),
    ("cryptography", ["cipher", "key", "hash", "signature", "protocol"]),
    ("genetics", ["DNA", "gene", "mutation", "CRISPR", "chromosome"]),
    ("robotics", ["actuator", "sensor", "kinematics", "SLAM", "controller"]),
    ("economics", ["inflation", "GDP", "market", "fiscal", "monetary"]),
    ("linguistics", ["syntax", "morphology", "phoneme", "semantics", "pragmatics"]),
]


class SyntheticMultiHopBenchmark(Benchmark):
    """Generates synthetic multi-hop retrieval problems."""

    def __init__(self, config: SyntheticConfig | None = None):
        self.config = config or SyntheticConfig()
        self._corpus: list[dict] = []
        self._samples: list[BenchmarkSample] = []
        self._loaded = False

    @property
    def name(self) -> str:
        return "synthetic-multihop"

    @property
    def description(self) -> str:
        return (
            f"Synthetic multi-hop benchmark: {self.config.num_chains} chains, "
            f"{self.config.hops_per_chain} hops, "
            f"{self.config.distractors_per_chain} distractors each"
        )

    def categories(self) -> list[str]:
        return [f"{h}-hop" for h in range(1, self.config.hops_per_chain + 1)]

    def load(self, data_dir: str | None = None) -> None:
        if self._loaded:
            return
        self._generate()
        self._loaded = True

    def _generate(self) -> None:
        rng = random.Random(self.config.seed)
        topics = list(_TOPICS)

        for chain_idx in range(self.config.num_chains):
            topic_name, keywords = topics[chain_idx % len(topics)]
            chain_docs = []
            facts = []

            # Generate chain documents (each depends on previous)
            for hop in range(self.config.hops_per_chain):
                doc_id = f"chain{chain_idx}_hop{hop}"
                kw1 = rng.choice(keywords)
                kw2 = rng.choice(keywords)
                fact = f"In {topic_name}, {kw1} is related to {kw2} at stage {hop + 1}"
                facts.append(fact)

                if hop == 0:
                    text = f"Research note on {topic_name}: {fact}. This is a foundational observation."
                else:
                    prev_fact = facts[hop - 1]
                    text = f"Follow-up on {topic_name}: Building on the finding that '{prev_fact}', we now observe: {fact}."

                chain_docs.append(doc_id)
                self._corpus.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": {"chain": chain_idx, "hop": hop, "topic": topic_name},
                })

            # Generate distractor documents
            for d in range(self.config.distractors_per_chain):
                doc_id = f"chain{chain_idx}_distractor{d}"
                other_topic, other_kw = rng.choice(topics)
                text = (
                    f"Note on {other_topic}: "
                    f"{rng.choice(other_kw)} and {rng.choice(other_kw)} "
                    f"are important considerations in this field."
                )
                self._corpus.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": {"chain": chain_idx, "distractor": True, "topic": other_topic},
                })

            # Generate questions at different hop depths
            for hop_depth in range(1, self.config.hops_per_chain + 1):
                q_id = f"chain{chain_idx}_q{hop_depth}hop"
                relevant_docs = chain_docs[:hop_depth]

                if hop_depth == 1:
                    question = f"What was the foundational observation about {topic_name}?"
                    answer = facts[0]
                else:
                    question = (
                        f"What is the {hop_depth}-step finding in {topic_name}, "
                        f"building from the initial observation?"
                    )
                    answer = facts[hop_depth - 1]

                self._samples.append(BenchmarkSample(
                    question_id=q_id,
                    question=question,
                    ground_truth=answer,
                    category=f"{hop_depth}-hop",
                    corpus_doc_ids=relevant_docs,
                    metadata={"chain": chain_idx, "hops_needed": hop_depth},
                ))

    def corpus(self) -> list[dict]:
        return self._corpus

    def samples(self) -> list[BenchmarkSample]:
        return self._samples
