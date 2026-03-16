"""Pluggable hypothesis implementations for RAG retrieval strategies.

A hypothesis modifies how retrieval results are selected, reranked,
or presented to the LLM. The arena tests each hypothesis against
the same benchmark and backend, producing comparable scorecards.

Hypotheses are organized into subpackages by technique:
  geometric/        — embedding-space reranking, diversity, spectral methods
  cross_encoder/    — cross-encoder reranking variants
  deep_pool/        — deep candidate pool + CE reranking
  llm/              — LLM-powered query decomposition, reranking, IRCoT
  multi_resolution/ — multi-resolution retrieval, advanced rerankers
  hybrid/           — hybrid composition experiments
"""

from .base import Hypothesis

# ── Geometric / embedding-space hypotheses ─────────────────────────
from .geometric.baseline_flat import FlatBaselineHypothesis
from .geometric.pca_diversity import PCADiversityHypothesis
from .geometric.pca_grouped import PCAGroupedHypothesis
from .geometric.dpp_selection import DPPSelectionHypothesis
from .geometric.convex_hull_coverage import ConvexHullCoverageHypothesis
from .geometric.centroid_drift import CentroidDriftHypothesis
from .geometric.spectral_gap import SpectralGapHypothesis
from .geometric.local_intrinsic_dimension import LocalIntrinsicDimensionHypothesis
from .geometric.hyde import HyDEHypothesis
from .geometric.cone_retrieval import ConeRetrievalHypothesis
from .geometric.geodesic_interpolation import GeodesicInterpolationHypothesis
from .geometric.anti_hubness import AntiHubnessHypothesis
from .geometric.mean_bias_correction import MeanBiasCorrectionHypothesis
from .geometric.query_decomposition import QueryDecompositionHypothesis
from .geometric.capacity_partition import CapacityPartitionHypothesis
from .geometric.score_calibration import ScoreCalibrationHypothesis
from .geometric.reciprocal_rank_fusion import RRFMultiPerspectiveHypothesis
from .geometric.hierarchical_cluster_retrieval import HierarchicalClusterHypothesis
from .geometric.graph_community_retrieval import GraphCommunityHypothesis
from .geometric.isotropy_enhancement import IsotropyEnhancementHypothesis
from .geometric.query_drift_correction import QueryDriftCorrectionHypothesis
from .geometric.adaptive_context_window import AdaptiveContextWindowHypothesis
from .geometric.cross_encoder_proxy import CrossEncoderProxyHypothesis
from .geometric.mahalanobis_retrieval import MahalanobisRetrievalHypothesis
from .geometric.density_peak_selection import DensityPeakSelectionHypothesis
from .geometric.information_bottleneck import InformationBottleneckHypothesis
from .geometric.spectral_reranking import SpectralRerankingHypothesis
from .geometric.leverage_score_sampling import LeverageScoreSamplingHypothesis
from .geometric.optimal_transport_reranking import OptimalTransportHypothesis
from .geometric.submodular_coverage import SubmodularCoverageHypothesis
from .geometric.topological_persistence import TopologicalPersistenceHypothesis
from .geometric.kernel_herding import KernelHerdingHypothesis
from .geometric.variance_reduction import VarianceReductionHypothesis
from .geometric.influence_function import InfluenceFunctionHypothesis
from .geometric.random_projection_ensemble import RandomProjectionEnsembleHypothesis
from .geometric.curvature_aware import CurvatureAwareHypothesis
from .geometric.anchor_expansion import AnchorExpansionHypothesis
from .geometric.mutual_information_reranking import MutualInformationHypothesis
from .geometric.embedding_triangulation import EmbeddingTriangulationHypothesis
from .geometric.csls_pure import PureCSLSHypothesis
from .geometric.residual_query import ResidualQueryHypothesis
from .geometric.spectral_query_decomp import SpectralQueryDecompHypothesis
from .geometric.embedding_gradient_ascent import EmbeddingGradientAscentHypothesis
from .geometric.void_detection import VoidDetectionHypothesis
from .geometric.relevance_field import RelevanceFieldHypothesis
from .geometric.contrastive_steering import ContrastiveSteeringHypothesis
from .geometric.expanded_retrieval import ExpandedRetrievalCEHypothesis

# ── Cross-encoder reranking ────────────────────────────────────────
from .cross_encoder.cross_encoder_reranker import CrossEncoderRerankerHypothesis
from .cross_encoder.ce_score_fusion import CEScoreFusionHypothesis
from .cross_encoder.ce_iterative import CEIterativeHypothesis
from .cross_encoder.ce_segmented import CESegmentedHypothesis
from .cross_encoder.ce_pairwise import CEPairwiseHypothesis
from .cross_encoder.ce_calibrated import CECalibratedHypothesis
from .cross_encoder.ce_diversity_bonus import CEDiversityBonusHypothesis
from .cross_encoder.ce_reciprocal_neighbor import CEReciprocalNeighborHypothesis
from .cross_encoder.ce_multi_window import CEMultiWindowHypothesis
from .cross_encoder.ce_title_boost import CETitleBoostHypothesis
from .cross_encoder.ce_ensemble import CEEnsembleHypothesis
from .cross_encoder.ce_multihop_iterative import CEMultihopIterativeHypothesis
from .cross_encoder.ce_title_multiwindow import CETitleMultiWindowHypothesis
from .cross_encoder.query_decomp_ce import QueryDecompCEHypothesis
from .cross_encoder.ce_larger_model import CELargerModelHypothesis
from .cross_encoder.ce_with_context import CEWithContextHypothesis
from .cross_encoder.ce_sentence_level import CESentenceLevelHypothesis
from .cross_encoder.ce_keyword_focused import CEKeywordFocusedHypothesis
from .cross_encoder.ce_answer_extraction import CEAnswerExtractionHypothesis
from .cross_encoder.bm25_boosted_ce import BM25BoostedCEHypothesis
from .cross_encoder.ce_negative_feedback import CENegativeFeedbackHypothesis
from .cross_encoder.ce_cross_doc import CECrossDocHypothesis
from .cross_encoder.ce_query_type_adaptive import CEQueryTypeAdaptiveHypothesis
from .cross_encoder.ce_coverage_greedy import CECoverageGreedyHypothesis
from .cross_encoder.bm25_text_feature_ce import BM25TextFeatureCEHypothesis
from .cross_encoder.temperature_scaled_ce import TemperatureScaledCEHypothesis
from .cross_encoder.csls_prefilter_ce import CSLSPrefilterCEHypothesis
from .cross_encoder.lid_gated_pool_ce import LIDGatedPoolCEHypothesis
from .cross_encoder.hub_aware_deep_pool_ce import HubAwareDeepPoolCEHypothesis
from .cross_encoder.routed_reranker import RoutedRerankerHypothesis
from .cross_encoder.cross_model_maxsim import CrossModelMaxSimHypothesis

# ── Deep candidate pool + CE ──────────────────────────────────────
from .deep_pool.deep_pool_ce import DeepPoolCEHypothesis
from .deep_pool.deep_pool_50_ce import DeepPool50CEHypothesis
from .deep_pool.deep_pool_100_ce import DeepPool100CEHypothesis
from .deep_pool.adaptive_pool_depth_ce import AdaptivePoolDepthCEHypothesis
from .deep_pool.two_stage_deep_pool_ce import TwoStageDeepPoolCEHypothesis
from .deep_pool.deep_pool_decomp_ce import DeepPoolDecompCEHypothesis
from .deep_pool.deep_pool_ircot_ce import DeepPoolIRCoTCEHypothesis
from .deep_pool.deep_pool_50_ce_l12 import DeepPool50CEL12Hypothesis

# ── LLM-powered retrieval ─────────────────────────────────────────
from .llm.llm_query_decomp_ce import LLMQueryDecompCEHypothesis
from .llm.ircot_simplified import IRCoTSimplifiedHypothesis
from .llm.ircot_full import IRCoTFullHypothesis
from .llm.llm_bridge_entity_ce import LLMBridgeEntityCEHypothesis
from .llm.llm_relevance_judge import LLMRelevanceJudgeHypothesis
from .llm.llm_query_expansion_ce import LLMQueryExpansionCEHypothesis
from .llm.llm_query_fusion_ce import LLMQueryFusionCEHypothesis
from .llm.llm_listwise_reranker import LLMListwiseRerankerHypothesis
from .llm.llm_pointwise_reranker import LLMPointwiseRerankerHypothesis

# ── Multi-resolution & advanced rerankers ─────────────────────────
from .multi_resolution.multi_resolution import MultiResolutionHypothesis
from .multi_resolution.multi_resolution_v2 import MultiResolutionV2Hypothesis
from .multi_resolution.mram_bge_reranker import MRAMBGERerankerHypothesis
from .multi_resolution.bge_reranker import BGERerankerHypothesis
try:
    from .multi_resolution.mxbai_reranker import MxbaiRerankerHypothesis
except ImportError:
    MxbaiRerankerHypothesis = None  # requires torch
from .multi_resolution.late_interaction_reranker import LateInteractionRerankerHypothesis
from .multi_resolution.late_interaction_mram import LateInteractionMRAMHypothesis
from .multi_resolution.multi_reranker_ensemble import MultiRerankerEnsembleHypothesis
from .multi_resolution.gated_mram_ce import GatedMRAMCEHypothesis

# ── Hybrid composition experiments ────────────────────────────────
from .hybrid.hybrid_antihub_influence import HybridAntihubInfluenceHypothesis
from .hybrid.hybrid_csls_topo_calibrated import HybridCSLSTopoCalibratedHypothesis
from .hybrid.hybrid_rrf_top5 import HybridRRFTop5Hypothesis

__all__ = [
    "Hypothesis",
    # Geometric / embedding-space
    "FlatBaselineHypothesis",
    "PCADiversityHypothesis",
    "PCAGroupedHypothesis",
    "DPPSelectionHypothesis",
    "ConvexHullCoverageHypothesis",
    "CentroidDriftHypothesis",
    "SpectralGapHypothesis",
    "LocalIntrinsicDimensionHypothesis",
    "HyDEHypothesis",
    "ConeRetrievalHypothesis",
    "GeodesicInterpolationHypothesis",
    "AntiHubnessHypothesis",
    "MeanBiasCorrectionHypothesis",
    "QueryDecompositionHypothesis",
    "CapacityPartitionHypothesis",
    "ScoreCalibrationHypothesis",
    "RRFMultiPerspectiveHypothesis",
    "HierarchicalClusterHypothesis",
    "GraphCommunityHypothesis",
    "IsotropyEnhancementHypothesis",
    "QueryDriftCorrectionHypothesis",
    "AdaptiveContextWindowHypothesis",
    "CrossEncoderProxyHypothesis",
    "MahalanobisRetrievalHypothesis",
    "DensityPeakSelectionHypothesis",
    "InformationBottleneckHypothesis",
    "SpectralRerankingHypothesis",
    "LeverageScoreSamplingHypothesis",
    "OptimalTransportHypothesis",
    "SubmodularCoverageHypothesis",
    "TopologicalPersistenceHypothesis",
    "KernelHerdingHypothesis",
    "VarianceReductionHypothesis",
    "InfluenceFunctionHypothesis",
    "RandomProjectionEnsembleHypothesis",
    "CurvatureAwareHypothesis",
    "AnchorExpansionHypothesis",
    "MutualInformationHypothesis",
    "EmbeddingTriangulationHypothesis",
    "PureCSLSHypothesis",
    "ResidualQueryHypothesis",
    "SpectralQueryDecompHypothesis",
    "EmbeddingGradientAscentHypothesis",
    "VoidDetectionHypothesis",
    "RelevanceFieldHypothesis",
    "ContrastiveSteeringHypothesis",
    "ExpandedRetrievalCEHypothesis",
    # Cross-encoder reranking
    "CrossEncoderRerankerHypothesis",
    "CEScoreFusionHypothesis",
    "CEIterativeHypothesis",
    "CESegmentedHypothesis",
    "CEPairwiseHypothesis",
    "CECalibratedHypothesis",
    "CEDiversityBonusHypothesis",
    "CEReciprocalNeighborHypothesis",
    "CEMultiWindowHypothesis",
    "CETitleBoostHypothesis",
    "CEEnsembleHypothesis",
    "CEMultihopIterativeHypothesis",
    "CETitleMultiWindowHypothesis",
    "QueryDecompCEHypothesis",
    "CELargerModelHypothesis",
    "CEWithContextHypothesis",
    "CESentenceLevelHypothesis",
    "CEKeywordFocusedHypothesis",
    "CEAnswerExtractionHypothesis",
    "BM25BoostedCEHypothesis",
    "CENegativeFeedbackHypothesis",
    "CECrossDocHypothesis",
    "CEQueryTypeAdaptiveHypothesis",
    "CECoverageGreedyHypothesis",
    "BM25TextFeatureCEHypothesis",
    "TemperatureScaledCEHypothesis",
    "CSLSPrefilterCEHypothesis",
    "LIDGatedPoolCEHypothesis",
    "HubAwareDeepPoolCEHypothesis",
    "RoutedRerankerHypothesis",
    "CrossModelMaxSimHypothesis",
    # Deep pool
    "DeepPoolCEHypothesis",
    "DeepPool50CEHypothesis",
    "DeepPool100CEHypothesis",
    "AdaptivePoolDepthCEHypothesis",
    "TwoStageDeepPoolCEHypothesis",
    "DeepPoolDecompCEHypothesis",
    "DeepPoolIRCoTCEHypothesis",
    "DeepPool50CEL12Hypothesis",
    # LLM-powered
    "LLMQueryDecompCEHypothesis",
    "IRCoTSimplifiedHypothesis",
    "IRCoTFullHypothesis",
    "LLMBridgeEntityCEHypothesis",
    "LLMRelevanceJudgeHypothesis",
    "LLMQueryExpansionCEHypothesis",
    "LLMQueryFusionCEHypothesis",
    "LLMListwiseRerankerHypothesis",
    "LLMPointwiseRerankerHypothesis",
    # Multi-resolution & advanced rerankers
    "MultiResolutionHypothesis",
    "MultiResolutionV2Hypothesis",
    "MRAMBGERerankerHypothesis",
    "BGERerankerHypothesis",
    "MxbaiRerankerHypothesis",
    "LateInteractionRerankerHypothesis",
    "LateInteractionMRAMHypothesis",
    "MultiRerankerEnsembleHypothesis",
    "GatedMRAMCEHypothesis",
    # Hybrid composition
    "HybridAntihubInfluenceHypothesis",
    "HybridCSLSTopoCalibratedHypothesis",
    "HybridRRFTop5Hypothesis",
]
