"""
Token sequence quality analysis for MIDI validation.

This module analyzes the quality of token sequences generated during tokenization,
including vocabulary coverage, sequence coherence, compression ratios, and
pattern detection.
"""

import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from midi_parser.core.tokenizer_manager import TokenizationResult
from midi_parser.config.defaults import TokenizerConfig

logger = logging.getLogger(__name__)


@dataclass
class VocabularyAnalysis:
    """Analysis of vocabulary usage and coverage."""
    total_vocabulary_size: int = 0
    used_vocabulary_size: int = 0
    coverage_ratio: float = 0.0
    token_frequency_distribution: Dict[int, int] = field(default_factory=dict)
    most_common_tokens: List[Tuple[int, int]] = field(default_factory=list)
    least_common_tokens: List[Tuple[int, int]] = field(default_factory=list)
    unused_token_count: int = 0
    token_entropy: float = 0.0
    vocabulary_efficiency: float = 0.0
    token_type_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class SequenceCoherenceAnalysis:
    """Analysis of token sequence coherence and structure."""
    sequence_length: int = 0
    unique_token_count: int = 0
    repetition_ratio: float = 0.0
    bigram_entropy: float = 0.0
    trigram_entropy: float = 0.0
    sequence_perplexity: float = 0.0
    pattern_consistency: float = 0.0
    structural_coherence: float = 0.0
    transition_smoothness: float = 0.0
    sequence_complexity: float = 0.0


@dataclass
class CompressionAnalysis:
    """Analysis of tokenization compression efficiency."""
    original_note_count: int = 0
    token_count: int = 0
    compression_ratio: float = 0.0
    bits_per_note: float = 0.0
    effective_compression: float = 0.0
    information_density: float = 0.0
    redundancy_factor: float = 0.0
    encoding_efficiency: float = 0.0


@dataclass
class PatternAnalysis:
    """Analysis of patterns in token sequences."""
    common_bigrams: List[Tuple[Tuple[int, int], int]] = field(default_factory=list)
    common_trigrams: List[Tuple[Tuple[int, int, int], int]] = field(default_factory=list)
    repeated_patterns: List[Dict[str, Any]] = field(default_factory=list)
    pattern_frequency: Dict[str, int] = field(default_factory=dict)
    musical_structure_detected: bool = False
    bar_pattern_regularity: float = 0.0
    phrase_structure_coherence: float = 0.0
    motif_preservation: float = 0.0


@dataclass
class SequenceQualityMetrics:
    """Comprehensive token sequence quality metrics."""
    vocabulary_analysis: VocabularyAnalysis = field(default_factory=VocabularyAnalysis)
    coherence_analysis: SequenceCoherenceAnalysis = field(default_factory=SequenceCoherenceAnalysis)
    compression_analysis: CompressionAnalysis = field(default_factory=CompressionAnalysis)
    pattern_analysis: PatternAnalysis = field(default_factory=PatternAnalysis)
    overall_sequence_quality: float = 0.0
    quality_scores: Dict[str, float] = field(default_factory=dict)
    quality_warnings: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "overall_sequence_quality": round(self.overall_sequence_quality, 4),
            "quality_scores": {k: round(v, 4) for k, v in self.quality_scores.items()},
            "vocabulary": {
                "total_size": self.vocabulary_analysis.total_vocabulary_size,
                "used_size": self.vocabulary_analysis.used_vocabulary_size,
                "coverage_ratio": round(self.vocabulary_analysis.coverage_ratio, 4),
                "entropy": round(self.vocabulary_analysis.token_entropy, 4),
                "efficiency": round(self.vocabulary_analysis.vocabulary_efficiency, 4)
            },
            "coherence": {
                "sequence_length": self.coherence_analysis.sequence_length,
                "unique_tokens": self.coherence_analysis.unique_token_count,
                "repetition_ratio": round(self.coherence_analysis.repetition_ratio, 4),
                "perplexity": round(self.coherence_analysis.sequence_perplexity, 4),
                "structural_coherence": round(self.coherence_analysis.structural_coherence, 4)
            },
            "compression": {
                "compression_ratio": round(self.compression_analysis.compression_ratio, 4),
                "bits_per_note": round(self.compression_analysis.bits_per_note, 4),
                "encoding_efficiency": round(self.compression_analysis.encoding_efficiency, 4)
            },
            "patterns": {
                "musical_structure_detected": self.pattern_analysis.musical_structure_detected,
                "bar_pattern_regularity": round(self.pattern_analysis.bar_pattern_regularity, 4),
                "phrase_coherence": round(self.pattern_analysis.phrase_structure_coherence, 4)
            },
            "warnings": self.quality_warnings,
            "suggestions": self.optimization_suggestions
        }


class SequenceQualityAnalyzer:
    """
    Analyzes the quality of token sequences generated during tokenization.
    
    This class provides comprehensive analysis of token sequences, including
    vocabulary utilization, sequence coherence, compression efficiency, and
    pattern detection.
    """
    
    def __init__(
        self,
        tokenizer_config: Optional[TokenizerConfig] = None,
        strategy: str = "REMI"
    ):
        """
        Initialize the sequence quality analyzer.
        
        Args:
            tokenizer_config: Tokenizer configuration
            strategy: Tokenization strategy
        """
        self.tokenizer_config = tokenizer_config
        self.strategy = strategy
        self.special_tokens = self._identify_special_tokens()
        
    def analyze_sequence_quality(
        self,
        tokenization_result: TokenizationResult,
        original_note_count: Optional[int] = None,
        vocabulary: Optional[Dict[str, int]] = None,
        token_types: Optional[Dict[int, str]] = None
    ) -> SequenceQualityMetrics:
        """
        Perform comprehensive token sequence quality analysis.
        
        Args:
            tokenization_result: Result from tokenization
            original_note_count: Number of notes in original MIDI
            vocabulary: Token vocabulary mapping
            token_types: Mapping of token IDs to their types
            
        Returns:
            SequenceQualityMetrics with comprehensive analysis
        """
        logger.info(f"Starting sequence quality analysis for {self.strategy}")
        
        metrics = SequenceQualityMetrics()
        
        # Get tokens
        tokens = tokenization_result.tokens
        vocab = vocabulary or tokenization_result.vocabulary
        
        # Analyze vocabulary usage
        metrics.vocabulary_analysis = self._analyze_vocabulary_usage(
            tokens, vocab, token_types
        )
        
        # Analyze sequence coherence
        metrics.coherence_analysis = self._analyze_sequence_coherence(tokens)
        
        # Analyze compression efficiency
        metrics.compression_analysis = self._analyze_compression(
            tokens, original_note_count, vocab
        )
        
        # Analyze patterns
        metrics.pattern_analysis = self._analyze_patterns(tokens, token_types)
        
        # Calculate overall quality score
        metrics.overall_sequence_quality = self._calculate_overall_quality(metrics)
        
        # Generate quality scores
        metrics.quality_scores = self._calculate_quality_scores(metrics)
        
        # Generate warnings and suggestions
        metrics.quality_warnings = self._generate_quality_warnings(metrics)
        metrics.optimization_suggestions = self._generate_optimization_suggestions(metrics)
        
        logger.info(f"Sequence quality analysis complete. Quality: {metrics.overall_sequence_quality:.2%}")
        
        return metrics
    
    def _identify_special_tokens(self) -> Set[str]:
        """
        Identify special tokens based on strategy.
        
        Returns:
            Set of special token patterns
        """
        special_tokens = {
            'REMI': {'Bar', 'Position', 'Tempo', 'TimeSignature', 'Rest'},
            'TSD': {'TimeShift', 'Rest'},
            'Structured': {'Bar', 'Beat', 'Position'},
            'CPWord': {'Family', 'Bar'},
            'Octuple': {'Time', 'Bar'}
        }
        
        return special_tokens.get(self.strategy, set())
    
    def _analyze_vocabulary_usage(
        self,
        tokens: List[int],
        vocabulary: Optional[Dict[str, int]],
        token_types: Optional[Dict[int, str]]
    ) -> VocabularyAnalysis:
        """
        Analyze vocabulary usage and coverage.
        
        Args:
            tokens: Token sequence
            vocabulary: Token vocabulary
            token_types: Token type mapping
            
        Returns:
            VocabularyAnalysis with results
        """
        analysis = VocabularyAnalysis()
        
        if not tokens:
            return analysis
        
        # Calculate vocabulary sizes
        if vocabulary:
            analysis.total_vocabulary_size = len(vocabulary)
        else:
            analysis.total_vocabulary_size = max(tokens) + 1 if tokens else 0
        
        # Token frequency distribution
        token_counts = Counter(tokens)
        analysis.token_frequency_distribution = dict(token_counts)
        analysis.used_vocabulary_size = len(token_counts)
        
        # Coverage ratio
        if analysis.total_vocabulary_size > 0:
            analysis.coverage_ratio = analysis.used_vocabulary_size / analysis.total_vocabulary_size
        
        # Most and least common tokens
        analysis.most_common_tokens = token_counts.most_common(20)
        if len(token_counts) > 20:
            analysis.least_common_tokens = token_counts.most_common()[-20:]
        
        # Unused tokens
        if vocabulary:
            used_tokens = set(tokens)
            all_tokens = set(vocabulary.values())
            analysis.unused_token_count = len(all_tokens - used_tokens)
        
        # Calculate token entropy
        analysis.token_entropy = self._calculate_entropy(list(token_counts.values()))
        
        # Calculate vocabulary efficiency
        if analysis.total_vocabulary_size > 0:
            optimal_size = len(token_counts)  # Ideal vocab size = unique tokens used
            analysis.vocabulary_efficiency = optimal_size / analysis.total_vocabulary_size
        
        # Analyze token type distribution if available
        if token_types:
            type_counts = defaultdict(int)
            for token in tokens:
                if token in token_types:
                    type_counts[token_types[token]] += 1
            analysis.token_type_distribution = dict(type_counts)
        
        return analysis
    
    def _analyze_sequence_coherence(self, tokens: List[int]) -> SequenceCoherenceAnalysis:
        """
        Analyze token sequence coherence and structure.
        
        Args:
            tokens: Token sequence
            
        Returns:
            SequenceCoherenceAnalysis with results
        """
        analysis = SequenceCoherenceAnalysis()
        
        if not tokens:
            return analysis
        
        analysis.sequence_length = len(tokens)
        analysis.unique_token_count = len(set(tokens))
        
        # Repetition ratio
        if analysis.sequence_length > 0:
            analysis.repetition_ratio = 1.0 - (analysis.unique_token_count / analysis.sequence_length)
        
        # Calculate n-gram entropies
        bigrams = self._get_ngrams(tokens, 2)
        trigrams = self._get_ngrams(tokens, 3)
        
        if bigrams:
            bigram_counts = Counter(bigrams)
            analysis.bigram_entropy = self._calculate_entropy(list(bigram_counts.values()))
        
        if trigrams:
            trigram_counts = Counter(trigrams)
            analysis.trigram_entropy = self._calculate_entropy(list(trigram_counts.values()))
        
        # Calculate sequence perplexity (simplified)
        analysis.sequence_perplexity = self._calculate_perplexity(tokens)
        
        # Pattern consistency
        analysis.pattern_consistency = self._calculate_pattern_consistency(tokens)
        
        # Structural coherence
        analysis.structural_coherence = self._calculate_structural_coherence(tokens)
        
        # Transition smoothness
        analysis.transition_smoothness = self._calculate_transition_smoothness(tokens)
        
        # Sequence complexity
        analysis.sequence_complexity = self._calculate_sequence_complexity(tokens)
        
        return analysis
    
    def _analyze_compression(
        self,
        tokens: List[int],
        original_note_count: Optional[int],
        vocabulary: Optional[Dict[str, int]]
    ) -> CompressionAnalysis:
        """
        Analyze tokenization compression efficiency.
        
        Args:
            tokens: Token sequence
            original_note_count: Original number of notes
            vocabulary: Token vocabulary
            
        Returns:
            CompressionAnalysis with results
        """
        analysis = CompressionAnalysis()
        
        if not tokens:
            return analysis
        
        analysis.token_count = len(tokens)
        
        if original_note_count:
            analysis.original_note_count = original_note_count
            
            # Compression ratio (notes to tokens)
            if original_note_count > 0:
                analysis.compression_ratio = analysis.token_count / original_note_count
            
            # Bits per note (assuming log2 vocabulary size bits per token)
            vocab_size = len(vocabulary) if vocabulary else max(tokens) + 1
            if vocab_size > 1 and original_note_count > 0:
                bits_per_token = np.log2(vocab_size)
                total_bits = analysis.token_count * bits_per_token
                analysis.bits_per_note = total_bits / original_note_count
            
            # Effective compression (compared to raw MIDI encoding)
            # Assume raw MIDI uses ~3 bytes per note (note on, note off, velocity)
            raw_bits = original_note_count * 24  # 3 bytes * 8 bits
            if raw_bits > 0 and analysis.bits_per_note > 0:
                token_bits = original_note_count * analysis.bits_per_note
                analysis.effective_compression = 1.0 - (token_bits / raw_bits)
        
        # Information density
        if tokens:
            unique_tokens = len(set(tokens))
            analysis.information_density = unique_tokens / len(tokens)
        
        # Redundancy factor
        analysis.redundancy_factor = 1.0 - analysis.information_density
        
        # Encoding efficiency (based on entropy vs actual encoding)
        token_counts = Counter(tokens)
        entropy = self._calculate_entropy(list(token_counts.values()))
        if vocabulary and entropy > 0:
            max_entropy = np.log2(len(vocabulary))
            analysis.encoding_efficiency = entropy / max_entropy if max_entropy > 0 else 0
        
        return analysis
    
    def _analyze_patterns(
        self,
        tokens: List[int],
        token_types: Optional[Dict[int, str]]
    ) -> PatternAnalysis:
        """
        Analyze patterns in token sequences.
        
        Args:
            tokens: Token sequence
            token_types: Token type mapping
            
        Returns:
            PatternAnalysis with results
        """
        analysis = PatternAnalysis()
        
        if not tokens:
            return analysis
        
        # Find common bigrams and trigrams
        bigrams = self._get_ngrams(tokens, 2)
        trigrams = self._get_ngrams(tokens, 3)
        
        if bigrams:
            bigram_counts = Counter(bigrams)
            analysis.common_bigrams = bigram_counts.most_common(20)
        
        if trigrams:
            trigram_counts = Counter(trigrams)
            analysis.common_trigrams = trigram_counts.most_common(20)
        
        # Find repeated patterns
        analysis.repeated_patterns = self._find_repeated_patterns(tokens)
        
        # Analyze pattern frequency
        for pattern_info in analysis.repeated_patterns:
            pattern_str = str(pattern_info['pattern'])
            analysis.pattern_frequency[pattern_str] = pattern_info['count']
        
        # Detect musical structure
        analysis.musical_structure_detected = self._detect_musical_structure(tokens, token_types)
        
        # Analyze bar pattern regularity (for REMI and similar)
        analysis.bar_pattern_regularity = self._analyze_bar_regularity(tokens, token_types)
        
        # Analyze phrase structure
        analysis.phrase_structure_coherence = self._analyze_phrase_structure(tokens)
        
        # Analyze motif preservation
        analysis.motif_preservation = self._analyze_motif_preservation(tokens)
        
        return analysis
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """
        Calculate Shannon entropy from count distribution.
        
        Args:
            counts: List of counts
            
        Returns:
            Entropy value
        """
        if not counts:
            return 0.0
        
        total = sum(counts)
        if total == 0:
            return 0.0
        
        probabilities = [c / total for c in counts]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
        return entropy
    
    def _get_ngrams(self, tokens: List[int], n: int) -> List[Tuple[int, ...]]:
        """
        Extract n-grams from token sequence.
        
        Args:
            tokens: Token sequence
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        if len(tokens) < n:
            return []
        
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _calculate_perplexity(self, tokens: List[int]) -> float:
        """
        Calculate simplified perplexity of token sequence.
        
        Args:
            tokens: Token sequence
            
        Returns:
            Perplexity value
        """
        if not tokens:
            return 0.0
        
        # Simplified perplexity based on bigram probabilities
        bigrams = self._get_ngrams(tokens, 2)
        if not bigrams:
            return 0.0
        
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(tokens)
        
        log_prob_sum = 0.0
        count = 0
        
        for bigram in bigrams:
            prev_token = bigram[0]
            curr_token = bigram[1]
            
            # Calculate conditional probability P(curr|prev)
            bigram_count = bigram_counts[bigram]
            prev_count = unigram_counts[prev_token]
            
            if prev_count > 0:
                prob = bigram_count / prev_count
                if prob > 0:
                    log_prob_sum += np.log2(prob)
                    count += 1
        
        if count == 0:
            return float('inf')
        
        avg_log_prob = log_prob_sum / count
        perplexity = 2 ** (-avg_log_prob)
        
        return perplexity
    
    def _calculate_pattern_consistency(self, tokens: List[int]) -> float:
        """
        Calculate consistency of patterns in sequence.
        
        Args:
            tokens: Token sequence
            
        Returns:
            Pattern consistency score (0-1)
        """
        if len(tokens) < 10:
            return 1.0
        
        # Look for consistent patterns in chunks
        chunk_size = len(tokens) // 10
        if chunk_size < 5:
            chunk_size = 5
        
        chunk_patterns = []
        for i in range(0, len(tokens) - chunk_size, chunk_size):
            chunk = tokens[i:i+chunk_size]
            # Create pattern signature
            pattern_sig = (len(set(chunk)), statistics.mean(chunk), statistics.stdev(chunk) if len(chunk) > 1 else 0)
            chunk_patterns.append(pattern_sig)
        
        if len(chunk_patterns) < 2:
            return 1.0
        
        # Calculate consistency based on pattern similarity
        consistencies = []
        for i in range(len(chunk_patterns) - 1):
            # Compare adjacent chunks
            p1, p2 = chunk_patterns[i], chunk_patterns[i+1]
            similarity = 1.0 - (abs(p1[0] - p2[0]) / max(p1[0], p2[0]) if max(p1[0], p2[0]) > 0 else 0)
            consistencies.append(similarity)
        
        return statistics.mean(consistencies) if consistencies else 1.0
    
    def _calculate_structural_coherence(self, tokens: List[int]) -> float:
        """
        Calculate structural coherence of token sequence.
        
        Args:
            tokens: Token sequence
            
        Returns:
            Structural coherence score (0-1)
        """
        if len(tokens) < 10:
            return 1.0
        
        # Analyze local vs global token distribution
        global_dist = Counter(tokens)
        
        # Split into segments and compare local distributions
        num_segments = min(10, len(tokens) // 20)
        if num_segments < 2:
            return 1.0
        
        segment_size = len(tokens) // num_segments
        coherence_scores = []
        
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size if i < num_segments - 1 else len(tokens)
            segment = tokens[start:end]
            
            local_dist = Counter(segment)
            
            # Compare local to global distribution
            coherence = self._compare_distributions(local_dist, global_dist)
            coherence_scores.append(coherence)
        
        return statistics.mean(coherence_scores) if coherence_scores else 1.0
    
    def _calculate_transition_smoothness(self, tokens: List[int]) -> float:
        """
        Calculate smoothness of token transitions.
        
        Args:
            tokens: Token sequence
            
        Returns:
            Transition smoothness score (0-1)
        """
        if len(tokens) < 2:
            return 1.0
        
        # Calculate transition differences
        transitions = [abs(tokens[i+1] - tokens[i]) for i in range(len(tokens) - 1)]
        
        if not transitions:
            return 1.0
        
        # Smooth transitions have small differences
        mean_transition = statistics.mean(transitions)
        max_transition = max(transitions)
        
        if max_transition == 0:
            return 1.0
        
        # Normalize and invert (smaller transitions = higher smoothness)
        smoothness = 1.0 - (mean_transition / max_transition)
        
        return max(0.0, min(1.0, smoothness))
    
    def _calculate_sequence_complexity(self, tokens: List[int]) -> float:
        """
        Calculate complexity of token sequence.
        
        Args:
            tokens: Token sequence
            
        Returns:
            Complexity score (0-1)
        """
        if not tokens:
            return 0.0
        
        # Combine multiple complexity measures
        unique_ratio = len(set(tokens)) / len(tokens)
        
        # Compression ratio (how well sequence compresses indicates complexity)
        bigrams = self._get_ngrams(tokens, 2)
        bigram_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        # Entropy-based complexity
        token_counts = Counter(tokens)
        entropy = self._calculate_entropy(list(token_counts.values()))
        max_entropy = np.log2(len(token_counts)) if len(token_counts) > 1 else 1
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine measures
        complexity = (unique_ratio * 0.3 + bigram_ratio * 0.3 + entropy_ratio * 0.4)
        
        return max(0.0, min(1.0, complexity))
    
    def _compare_distributions(self, dist1: Counter, dist2: Counter) -> float:
        """
        Compare two distributions using cosine similarity.
        
        Args:
            dist1: First distribution
            dist2: Second distribution
            
        Returns:
            Similarity score (0-1)
        """
        if not dist1 or not dist2:
            return 0.0
        
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        v1 = [dist1.get(k, 0) for k in all_keys]
        v2 = [dist2.get(k, 0) for k in all_keys]
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        similarity = dot_product / (mag1 * mag2)
        return max(0.0, min(1.0, similarity))
    
    def _find_repeated_patterns(self, tokens: List[int], min_length: int = 4, min_count: int = 2) -> List[Dict[str, Any]]:
        """
        Find repeated patterns in token sequence.
        
        Args:
            tokens: Token sequence
            min_length: Minimum pattern length
            min_count: Minimum repetition count
            
        Returns:
            List of repeated pattern information
        """
        patterns = []
        seen_patterns = set()
        
        for length in range(min_length, min(20, len(tokens) // 2)):
            pattern_counts = defaultdict(list)
            
            for i in range(len(tokens) - length + 1):
                pattern = tuple(tokens[i:i+length])
                pattern_counts[pattern].append(i)
            
            for pattern, positions in pattern_counts.items():
                if len(positions) >= min_count and pattern not in seen_patterns:
                    patterns.append({
                        'pattern': pattern,
                        'length': length,
                        'count': len(positions),
                        'positions': positions[:10]  # Keep first 10 positions
                    })
                    seen_patterns.add(pattern)
        
        # Sort by count * length (favor longer, more frequent patterns)
        patterns.sort(key=lambda x: x['count'] * x['length'], reverse=True)
        
        return patterns[:20]  # Return top 20 patterns
    
    def _detect_musical_structure(self, tokens: List[int], token_types: Optional[Dict[int, str]]) -> bool:
        """
        Detect if tokens represent coherent musical structure.
        
        Args:
            tokens: Token sequence
            token_types: Token type mapping
            
        Returns:
            Whether musical structure is detected
        """
        if not tokens:
            return False
        
        # Check for presence of structural markers
        if token_types:
            structural_markers = ['Bar', 'Position', 'TimeSignature', 'Tempo']
            token_type_set = set(token_types.values())
            
            has_markers = any(marker in token_type_set for marker in structural_markers)
            if not has_markers:
                return False
        
        # Check for regular patterns that indicate structure
        # Look for recurring patterns at expected intervals
        expected_bar_length = len(tokens) // 16  # Assume at least 16 bars
        if expected_bar_length < 10:
            expected_bar_length = 10
        
        # Check for pattern regularity
        patterns = self._find_repeated_patterns(tokens, min_length=4, min_count=4)
        
        return len(patterns) > 5  # At least 5 repeated patterns indicates structure
    
    def _analyze_bar_regularity(self, tokens: List[int], token_types: Optional[Dict[int, str]]) -> float:
        """
        Analyze regularity of bar patterns (for strategies with bar tokens).
        
        Args:
            tokens: Token sequence
            token_types: Token type mapping
            
        Returns:
            Bar pattern regularity score (0-1)
        """
        if not tokens or not token_types:
            return 0.5  # Neutral score if can't analyze
        
        # Find bar tokens
        bar_positions = []
        for i, token in enumerate(tokens):
            if token in token_types and 'bar' in token_types[token].lower():
                bar_positions.append(i)
        
        if len(bar_positions) < 2:
            return 0.5  # Can't analyze with fewer than 2 bars
        
        # Calculate bar lengths
        bar_lengths = [bar_positions[i+1] - bar_positions[i] 
                      for i in range(len(bar_positions) - 1)]
        
        if not bar_lengths:
            return 0.5
        
        # Calculate regularity based on consistency of bar lengths
        mean_length = statistics.mean(bar_lengths)
        if mean_length == 0:
            return 0.0
        
        # Calculate coefficient of variation
        std_dev = statistics.stdev(bar_lengths) if len(bar_lengths) > 1 else 0
        cv = std_dev / mean_length if mean_length > 0 else 0
        
        # Convert to regularity score (lower CV = higher regularity)
        regularity = max(0.0, 1.0 - cv)
        
        return regularity
    
    def _analyze_phrase_structure(self, tokens: List[int]) -> float:
        """
        Analyze phrase structure coherence.
        
        Args:
            tokens: Token sequence
            
        Returns:
            Phrase structure coherence score (0-1)
        """
        if len(tokens) < 32:
            return 1.0  # Too short to analyze phrases
        
        # Typical phrase lengths in tokens (approximate)
        typical_phrase_lengths = [16, 32, 64, 128]
        
        # Check for patterns at phrase boundaries
        coherence_scores = []
        
        for phrase_length in typical_phrase_lengths:
            if len(tokens) < phrase_length * 2:
                continue
            
            # Check for similar patterns at phrase boundaries
            phrase_patterns = []
            for i in range(0, len(tokens) - phrase_length, phrase_length):
                # Get pattern at phrase boundary
                boundary_pattern = tokens[i:i+8]  # Look at 8 tokens at boundary
                phrase_patterns.append(tuple(boundary_pattern))
            
            if len(phrase_patterns) > 1:
                # Check pattern similarity
                pattern_counts = Counter(phrase_patterns)
                # High repetition indicates good phrase structure
                max_count = max(pattern_counts.values())
                coherence = max_count / len(phrase_patterns)
                coherence_scores.append(coherence)
        
        return statistics.mean(coherence_scores) if coherence_scores else 0.5
    
    def _analyze_motif_preservation(self, tokens: List[int]) -> float:
        """
        Analyze preservation of musical motifs.
        
        Args:
            tokens: Token sequence
            
        Returns:
            Motif preservation score (0-1)
        """
        if len(tokens) < 20:
            return 1.0
        
        # Look for short repeated patterns (motifs)
        motif_patterns = self._find_repeated_patterns(tokens, min_length=3, min_count=3)
        
        if not motif_patterns:
            return 0.5  # No clear motifs
        
        # Calculate motif coverage
        total_motif_tokens = sum(p['count'] * p['length'] for p in motif_patterns[:10])
        coverage = total_motif_tokens / len(tokens)
        
        # Good motif preservation shows moderate coverage (not too much, not too little)
        if coverage < 0.1:
            score = coverage * 5  # Scale up low coverage
        elif coverage > 0.7:
            score = 1.0 - (coverage - 0.7)  # Penalize over-repetition
        else:
            score = 1.0  # Optimal range
        
        return max(0.0, min(1.0, score))
    
    def _calculate_overall_quality(self, metrics: SequenceQualityMetrics) -> float:
        """
        Calculate overall sequence quality score.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            Overall quality score (0-1)
        """
        # Weight different aspects
        weights = {
            'vocabulary': 0.2,
            'coherence': 0.3,
            'compression': 0.2,
            'patterns': 0.3
        }
        
        scores = {
            'vocabulary': (
                metrics.vocabulary_analysis.coverage_ratio * 0.3 +
                metrics.vocabulary_analysis.vocabulary_efficiency * 0.4 +
                min(1.0, metrics.vocabulary_analysis.token_entropy / 5.0) * 0.3  # Normalize entropy
            ),
            'coherence': (
                (1.0 - metrics.coherence_analysis.repetition_ratio) * 0.2 +
                metrics.coherence_analysis.structural_coherence * 0.3 +
                metrics.coherence_analysis.pattern_consistency * 0.2 +
                metrics.coherence_analysis.transition_smoothness * 0.3
            ),
            'compression': (
                min(1.0, 1.0 / max(metrics.compression_analysis.compression_ratio, 0.1)) * 0.3 +
                metrics.compression_analysis.encoding_efficiency * 0.4 +
                metrics.compression_analysis.information_density * 0.3
            ),
            'patterns': (
                (1.0 if metrics.pattern_analysis.musical_structure_detected else 0.5) * 0.3 +
                metrics.pattern_analysis.bar_pattern_regularity * 0.2 +
                metrics.pattern_analysis.phrase_structure_coherence * 0.25 +
                metrics.pattern_analysis.motif_preservation * 0.25
            )
        }
        
        overall = sum(scores[k] * weights[k] for k in weights)
        return max(0.0, min(1.0, overall))
    
    def _calculate_quality_scores(self, metrics: SequenceQualityMetrics) -> Dict[str, float]:
        """
        Calculate individual quality scores.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            Dictionary of quality scores
        """
        return {
            'vocabulary_coverage': metrics.vocabulary_analysis.coverage_ratio,
            'vocabulary_efficiency': metrics.vocabulary_analysis.vocabulary_efficiency,
            'token_entropy': min(1.0, metrics.vocabulary_analysis.token_entropy / 7.0),
            'sequence_coherence': metrics.coherence_analysis.structural_coherence,
            'pattern_consistency': metrics.coherence_analysis.pattern_consistency,
            'transition_smoothness': metrics.coherence_analysis.transition_smoothness,
            'compression_efficiency': min(1.0, 1.0 / max(metrics.compression_analysis.compression_ratio, 0.1)),
            'encoding_efficiency': metrics.compression_analysis.encoding_efficiency,
            'information_density': metrics.compression_analysis.information_density,
            'musical_structure': 1.0 if metrics.pattern_analysis.musical_structure_detected else 0.5,
            'bar_regularity': metrics.pattern_analysis.bar_pattern_regularity,
            'phrase_coherence': metrics.pattern_analysis.phrase_structure_coherence,
            'motif_preservation': metrics.pattern_analysis.motif_preservation
        }
    
    def _generate_quality_warnings(self, metrics: SequenceQualityMetrics) -> List[str]:
        """
        Generate warnings about sequence quality issues.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Vocabulary warnings
        if metrics.vocabulary_analysis.coverage_ratio < 0.1:
            warnings.append(f"Very low vocabulary usage: {metrics.vocabulary_analysis.coverage_ratio:.1%}")
        elif metrics.vocabulary_analysis.coverage_ratio > 0.9:
            warnings.append(f"Very high vocabulary usage may indicate inefficiency: {metrics.vocabulary_analysis.coverage_ratio:.1%}")
        
        if metrics.vocabulary_analysis.vocabulary_efficiency < 0.3:
            warnings.append(f"Low vocabulary efficiency: {metrics.vocabulary_analysis.vocabulary_efficiency:.1%}")
        
        if metrics.vocabulary_analysis.token_entropy < 2.0:
            warnings.append(f"Low token entropy indicates poor diversity: {metrics.vocabulary_analysis.token_entropy:.2f}")
        
        # Coherence warnings
        if metrics.coherence_analysis.repetition_ratio > 0.8:
            warnings.append(f"High repetition ratio: {metrics.coherence_analysis.repetition_ratio:.1%}")
        
        if metrics.coherence_analysis.structural_coherence < 0.5:
            warnings.append(f"Poor structural coherence: {metrics.coherence_analysis.structural_coherence:.2f}")
        
        if metrics.coherence_analysis.sequence_perplexity > 100:
            warnings.append(f"High perplexity indicates unpredictable sequence: {metrics.coherence_analysis.sequence_perplexity:.1f}")
        
        # Compression warnings
        if metrics.compression_analysis.compression_ratio > 3.0:
            warnings.append(f"Poor compression ratio: {metrics.compression_analysis.compression_ratio:.2f} tokens per note")
        
        if metrics.compression_analysis.encoding_efficiency < 0.5:
            warnings.append(f"Low encoding efficiency: {metrics.compression_analysis.encoding_efficiency:.1%}")
        
        # Pattern warnings
        if not metrics.pattern_analysis.musical_structure_detected:
            warnings.append("No clear musical structure detected in token sequence")
        
        if metrics.pattern_analysis.bar_pattern_regularity < 0.5:
            warnings.append(f"Irregular bar patterns: {metrics.pattern_analysis.bar_pattern_regularity:.1%}")
        
        if metrics.pattern_analysis.phrase_structure_coherence < 0.3:
            warnings.append(f"Poor phrase structure: {metrics.pattern_analysis.phrase_structure_coherence:.1%}")
        
        return warnings
    
    def _generate_optimization_suggestions(self, metrics: SequenceQualityMetrics) -> List[str]:
        """
        Generate optimization suggestions based on analysis.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Vocabulary optimization
        if metrics.vocabulary_analysis.coverage_ratio < 0.2:
            suggestions.append("Consider reducing vocabulary size - many tokens are unused")
        elif metrics.vocabulary_analysis.coverage_ratio > 0.95:
            suggestions.append("Consider expanding vocabulary to better capture musical nuances")
        
        if metrics.vocabulary_analysis.unused_token_count > metrics.vocabulary_analysis.total_vocabulary_size * 0.5:
            suggestions.append(f"Remove {metrics.vocabulary_analysis.unused_token_count} unused tokens from vocabulary")
        
        # Compression optimization
        if metrics.compression_analysis.compression_ratio > 2.5:
            suggestions.append("Consider using a more efficient tokenization strategy (e.g., CPWord for compression)")
        elif metrics.compression_analysis.compression_ratio < 0.5:
            suggestions.append("Token sequence may be too compressed - consider more granular encoding")
        
        if metrics.compression_analysis.redundancy_factor > 0.7:
            suggestions.append("High redundancy detected - consider using variable-length encoding")
        
        # Coherence optimization
        if metrics.coherence_analysis.repetition_ratio > 0.7:
            suggestions.append("Reduce repetition by adjusting tokenization parameters")
        
        if metrics.coherence_analysis.sequence_complexity < 0.3:
            suggestions.append("Sequence lacks complexity - check if musical nuances are being captured")
        elif metrics.coherence_analysis.sequence_complexity > 0.9:
            suggestions.append("Sequence too complex - consider simplifying tokenization parameters")
        
        # Pattern optimization
        if not metrics.pattern_analysis.musical_structure_detected:
            suggestions.append("Enable structural tokens (Bar, Position) for better musical representation")
        
        if metrics.pattern_analysis.bar_pattern_regularity < 0.6:
            suggestions.append("Adjust beat resolution to better capture rhythmic patterns")
        
        if len(metrics.pattern_analysis.repeated_patterns) < 3:
            suggestions.append("Few repeated patterns detected - verify motif preservation")
        
        # Strategy-specific suggestions
        if self.strategy == "REMI" and metrics.compression_analysis.compression_ratio > 2.0:
            suggestions.append("Consider CPWord strategy for better compression")
        elif self.strategy == "TSD" and metrics.coherence_analysis.structural_coherence < 0.6:
            suggestions.append("Consider REMI or Structured strategy for better structural representation")
        elif self.strategy == "CPWord" and metrics.pattern_analysis.phrase_structure_coherence < 0.5:
            suggestions.append("Consider REMI strategy for better phrase structure preservation")
        
        return suggestions