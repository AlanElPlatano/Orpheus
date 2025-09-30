"""
Advanced statistical similarity measures for MIDI comparison.

This module provides sophisticated statistical analysis methods for comparing
original and reconstructed MIDI files, including distribution comparisons,
correlation analysis, statistical significance testing, and outlier detection.
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, chi2_contingency, pearsonr, spearmanr
from miditoolkit import MidiFile

from parser.validation.validation_metrics import RoundTripMetrics

logger = logging.getLogger(__name__)


@dataclass
class DistributionComparison:
    """Statistical comparison of distributions."""
    ks_statistic: float = 0.0
    ks_pvalue: float = 0.0
    wasserstein_distance: float = 0.0
    jensen_shannon_divergence: float = 0.0
    hellinger_distance: float = 0.0
    total_variation_distance: float = 0.0
    chi_square_statistic: float = 0.0
    chi_square_pvalue: float = 0.0
    distribution_similarity_score: float = 0.0
    distributions_identical: bool = False


@dataclass
class CorrelationAnalysis:
    """Correlation analysis between musical features."""
    pitch_correlation: float = 0.0
    pitch_correlation_pvalue: float = 0.0
    velocity_correlation: float = 0.0
    velocity_correlation_pvalue: float = 0.0
    duration_correlation: float = 0.0
    duration_correlation_pvalue: float = 0.0
    onset_correlation: float = 0.0
    onset_correlation_pvalue: float = 0.0
    spearman_pitch_correlation: float = 0.0
    spearman_velocity_correlation: float = 0.0
    kendall_tau: float = 0.0
    mutual_information: float = 0.0
    cross_correlation_peak: float = 0.0
    lag_at_peak: int = 0


@dataclass
class SignificanceTests:
    """Statistical significance test results."""
    mann_whitney_pitch: Tuple[float, float] = (0.0, 0.0)  # statistic, p-value
    mann_whitney_velocity: Tuple[float, float] = (0.0, 0.0)
    mann_whitney_duration: Tuple[float, float] = (0.0, 0.0)
    wilcoxon_signed_rank: Tuple[float, float] = (0.0, 0.0)
    friedman_test: Tuple[float, float] = (0.0, 0.0)
    kruskal_wallis: Tuple[float, float] = (0.0, 0.0)
    paired_t_test: Tuple[float, float] = (0.0, 0.0)
    f_test_variance: Tuple[float, float] = (0.0, 0.0)
    significant_difference: bool = False
    confidence_level: float = 0.95


@dataclass
class OutlierAnalysis:
    """Outlier detection and analysis."""
    outlier_notes_original: List[int] = field(default_factory=list)
    outlier_notes_reconstructed: List[int] = field(default_factory=list)
    outlier_ratio_original: float = 0.0
    outlier_ratio_reconstructed: float = 0.0
    outlier_preservation_rate: float = 0.0
    zscore_outliers_original: int = 0
    zscore_outliers_reconstructed: int = 0
    iqr_outliers_original: int = 0
    iqr_outliers_reconstructed: int = 0
    isolation_forest_anomalies: int = 0
    local_outlier_factor: float = 0.0


@dataclass
class TemporalAnalysis:
    """Temporal statistical analysis."""
    autocorrelation_original: List[float] = field(default_factory=list)
    autocorrelation_reconstructed: List[float] = field(default_factory=list)
    partial_autocorrelation_diff: float = 0.0
    stationarity_original: bool = False
    stationarity_reconstructed: bool = False
    trend_similarity: float = 0.0
    seasonality_similarity: float = 0.0
    time_series_distance: float = 0.0
    dynamic_time_warping_distance: float = 0.0
    granger_causality_pvalue: float = 0.0


@dataclass
class StatisticalMetrics:
    """Comprehensive statistical analysis metrics."""
    distribution_comparison: Dict[str, DistributionComparison] = field(default_factory=dict)
    correlation_analysis: CorrelationAnalysis = field(default_factory=CorrelationAnalysis)
    significance_tests: SignificanceTests = field(default_factory=SignificanceTests)
    outlier_analysis: OutlierAnalysis = field(default_factory=OutlierAnalysis)
    temporal_analysis: TemporalAnalysis = field(default_factory=TemporalAnalysis)
    overall_statistical_similarity: float = 0.0
    statistical_scores: Dict[str, float] = field(default_factory=dict)
    statistical_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "overall_statistical_similarity": round(self.overall_statistical_similarity, 4),
            "statistical_scores": {k: round(v, 4) for k, v in self.statistical_scores.items()},
            "distributions": {
                name: {
                    "ks_statistic": round(comp.ks_statistic, 4),
                    "ks_pvalue": round(comp.ks_pvalue, 4),
                    "wasserstein_distance": round(comp.wasserstein_distance, 4),
                    "similarity_score": round(comp.distribution_similarity_score, 4)
                }
                for name, comp in self.distribution_comparison.items()
            },
            "correlations": {
                "pitch": round(self.correlation_analysis.pitch_correlation, 4),
                "velocity": round(self.correlation_analysis.velocity_correlation, 4),
                "duration": round(self.correlation_analysis.duration_correlation, 4),
                "mutual_information": round(self.correlation_analysis.mutual_information, 4)
            },
            "significance": {
                "significant_difference": self.significance_tests.significant_difference,
                "confidence_level": self.significance_tests.confidence_level
            },
            "outliers": {
                "outlier_ratio_original": round(self.outlier_analysis.outlier_ratio_original, 4),
                "outlier_ratio_reconstructed": round(self.outlier_analysis.outlier_ratio_reconstructed, 4),
                "outlier_preservation": round(self.outlier_analysis.outlier_preservation_rate, 4)
            },
            "temporal": {
                "stationarity_preserved": (self.temporal_analysis.stationarity_original == 
                                         self.temporal_analysis.stationarity_reconstructed),
                "trend_similarity": round(self.temporal_analysis.trend_similarity, 4),
                "dtw_distance": round(self.temporal_analysis.dynamic_time_warping_distance, 4)
            },
            "warnings": self.statistical_warnings
        }


class StatisticalComparator:
    """
    Performs advanced statistical comparison between MIDI files.
    
    This class provides sophisticated statistical analysis methods beyond
    basic comparison, including distribution tests, correlation analysis,
    significance testing, and outlier detection.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the statistical comparator.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def perform_statistical_comparison(
        self,
        original: MidiFile,
        reconstructed: MidiFile,
        round_trip_metrics: Optional[RoundTripMetrics] = None,
        detailed: bool = True
    ) -> StatisticalMetrics:
        """
        Perform comprehensive statistical comparison.
        
        Args:
            original: Original MIDI file
            reconstructed: Reconstructed MIDI file
            round_trip_metrics: Optional round-trip validation metrics
            detailed: Whether to perform detailed analysis
            
        Returns:
            StatisticalMetrics with comprehensive analysis
        """
        logger.info("Starting statistical comparison analysis")
        
        metrics = StatisticalMetrics()
        
        # Extract features from both MIDI files
        orig_features = self._extract_features(original)
        recon_features = self._extract_features(reconstructed)
        
        # Compare distributions
        metrics.distribution_comparison = self._compare_distributions(
            orig_features, recon_features, detailed
        )
        
        # Analyze correlations
        metrics.correlation_analysis = self._analyze_correlations(
            orig_features, recon_features
        )
        
        # Perform significance tests
        metrics.significance_tests = self._perform_significance_tests(
            orig_features, recon_features
        )
        
        # Detect outliers
        metrics.outlier_analysis = self._analyze_outliers(
            orig_features, recon_features
        )
        
        # Temporal analysis
        if detailed:
            metrics.temporal_analysis = self._analyze_temporal_patterns(
                orig_features, recon_features
            )
        
        # Calculate overall similarity
        metrics.overall_statistical_similarity = self._calculate_overall_similarity(metrics)
        
        # Generate statistical scores
        metrics.statistical_scores = self._calculate_statistical_scores(metrics)
        
        # Generate warnings
        metrics.statistical_warnings = self._generate_statistical_warnings(metrics)
        
        logger.info(f"Statistical comparison complete. Similarity: {metrics.overall_statistical_similarity:.2%}")
        
        return metrics
    
    def _extract_features(self, midi: MidiFile) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from MIDI file.
        
        Args:
            midi: MIDI file
            
        Returns:
            Dictionary of feature arrays
        """
        features = {
            'pitches': [],
            'velocities': [],
            'durations': [],
            'onsets': [],
            'intervals': [],
            'ioi': []  # Inter-onset intervals
        }
        
        all_notes = []
        for track in midi.instruments:
            all_notes.extend(track.notes)
        
        if not all_notes:
            return {k: np.array([]) for k in features}
        
        # Sort by onset time
        all_notes.sort(key=lambda n: n.start)
        
        for i, note in enumerate(all_notes):
            features['pitches'].append(note.pitch)
            features['velocities'].append(note.velocity)
            features['durations'].append(note.end - note.start)
            features['onsets'].append(note.start)
            
            # Calculate melodic intervals
            if i > 0:
                interval = note.pitch - all_notes[i-1].pitch
                features['intervals'].append(interval)
                ioi = note.start - all_notes[i-1].start
                features['ioi'].append(ioi)
        
        # Convert to numpy arrays
        return {k: np.array(v) for k, v in features.items()}
    
    def _compare_distributions(
        self,
        orig_features: Dict[str, np.ndarray],
        recon_features: Dict[str, np.ndarray],
        detailed: bool
    ) -> Dict[str, DistributionComparison]:
        """
        Compare feature distributions using multiple statistical tests.
        
        Args:
            orig_features: Original features
            recon_features: Reconstructed features
            detailed: Whether to perform detailed comparison
            
        Returns:
            Dictionary of distribution comparisons
        """
        comparisons = {}
        
        for feature_name in ['pitches', 'velocities', 'durations', 'intervals']:
            orig_data = orig_features.get(feature_name, np.array([]))
            recon_data = recon_features.get(feature_name, np.array([]))

            # Scipy functions might crash with insufficient data in any of the arrays
            # Checking this also prevents divisions by zero
            if len(orig_data) < 10 or len(recon_data) < 10:
                logger.warning(f"Insufficient data for {feature_name} comparison")
                return DistributionComparison()  # Return default
            
            comparison = DistributionComparison()

            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = ks_2samp(orig_data, recon_data)
            comparison.ks_statistic = ks_stat
            comparison.ks_pvalue = ks_pval
            comparison.distributions_identical = ks_pval > self.alpha
            
            # Wasserstein distance
            comparison.wasserstein_distance = stats.wasserstein_distance(
                orig_data, recon_data
            )
            
            if detailed:
                # Jensen-Shannon divergence
                comparison.jensen_shannon_divergence = self._calculate_js_divergence(
                    orig_data, recon_data
                )
                
                # Hellinger distance
                comparison.hellinger_distance = self._calculate_hellinger_distance(
                    orig_data, recon_data
                )
                
                # Total variation distance
                comparison.total_variation_distance = self._calculate_tv_distance(
                    orig_data, recon_data
                )
                
                # Chi-square test (for categorical data)
                if feature_name in ['pitches', 'velocities']:
                    chi2_stat, chi2_pval = self._perform_chi_square_test(
                        orig_data, recon_data
                    )
                    comparison.chi_square_statistic = chi2_stat
                    comparison.chi_square_pvalue = chi2_pval
            
            # Calculate overall distribution similarity
            comparison.distribution_similarity_score = self._calculate_distribution_similarity(
                comparison
            )
            
            # Put the result of the comparison in the array
            comparisons[feature_name] = comparison
        
        return comparisons

    # Prevents division by zero
    def safe_divide(numerator, denominator, default=0.0):
        """Safely divide with fallback for zero denominator."""
        return numerator / denominator if denominator != 0 else default
    
    def _analyze_correlations(
        self,
        orig_features: Dict[str, np.ndarray],
        recon_features: Dict[str, np.ndarray]
    ) -> CorrelationAnalysis:
        """
        Analyze correlations between original and reconstructed features.
        
        Args:
            orig_features: Original features
            recon_features: Reconstructed features
            
        Returns:
            CorrelationAnalysis with results
        """
        analysis = CorrelationAnalysis()
        
        # Match sequences for correlation (use minimum length)
        for feature_name in ['pitches', 'velocities', 'durations', 'onsets']:
            orig_data = orig_features.get(feature_name, np.array([]))
            recon_data = recon_features.get(feature_name, np.array([]))
            
            if len(orig_data) > 0 and len(recon_data) > 0:
                min_len = min(len(orig_data), len(recon_data))
                orig_matched = orig_data[:min_len]
                recon_matched = recon_data[:min_len]
                
                if min_len > 2:  # Need at least 3 points for correlation
                    # Pearson correlation
                    if feature_name == 'pitches':
                        r, p = pearsonr(orig_matched, recon_matched)
                        analysis.pitch_correlation = r
                        analysis.pitch_correlation_pvalue = p
                        
                        # Spearman correlation (rank-based)
                        rho, _ = spearmanr(orig_matched, recon_matched)
                        analysis.spearman_pitch_correlation = rho
                        
                    elif feature_name == 'velocities':
                        r, p = pearsonr(orig_matched, recon_matched)
                        analysis.velocity_correlation = r
                        analysis.velocity_correlation_pvalue = p
                        
                        rho, _ = spearmanr(orig_matched, recon_matched)
                        analysis.spearman_velocity_correlation = rho
                        
                    elif feature_name == 'durations':
                        r, p = pearsonr(orig_matched, recon_matched)
                        analysis.duration_correlation = r
                        analysis.duration_correlation_pvalue = p
                        
                    elif feature_name == 'onsets':
                        r, p = pearsonr(orig_matched, recon_matched)
                        analysis.onset_correlation = r
                        analysis.onset_correlation_pvalue = p
        
        # Calculate Kendall's tau (concordance)
        if len(orig_features.get('pitches', [])) > 0 and len(recon_features.get('pitches', [])) > 0:
            min_len = min(len(orig_features['pitches']), len(recon_features['pitches']))
            if min_len > 2:
                tau, _ = stats.kendalltau(
                    orig_features['pitches'][:min_len],
                    recon_features['pitches'][:min_len]
                )
                analysis.kendall_tau = tau
        
        # Calculate mutual information
        analysis.mutual_information = self._calculate_mutual_information(
            orig_features, recon_features
        )
        
        # Cross-correlation for lag detection
        if len(orig_features.get('pitches', [])) > 10 and len(recon_features.get('pitches', [])) > 10:
            peak, lag = self._calculate_cross_correlation(
                orig_features['pitches'], recon_features['pitches']
            )
            analysis.cross_correlation_peak = peak
            analysis.lag_at_peak = lag
        
        return analysis
    
    def _perform_significance_tests(
        self,
        orig_features: Dict[str, np.ndarray],
        recon_features: Dict[str, np.ndarray]
    ) -> SignificanceTests:
        """
        Perform statistical significance tests.
        
        Args:
            orig_features: Original features
            recon_features: Reconstructed features
            
        Returns:
            SignificanceTests with results
        """
        tests = SignificanceTests(confidence_level=self.confidence_level)
        
        # Mann-Whitney U tests (non-parametric)
        for feature_name, attr_name in [
            ('pitches', 'mann_whitney_pitch'),
            ('velocities', 'mann_whitney_velocity'),
            ('durations', 'mann_whitney_duration')
        ]:
            orig_data = orig_features.get(feature_name, np.array([]))
            recon_data = recon_features.get(feature_name, np.array([]))
            
            if len(orig_data) > 0 and len(recon_data) > 0:
                try:
                    u_stat, p_val = mannwhitneyu(orig_data, recon_data, alternative='two-sided')
                    setattr(tests, attr_name, (u_stat, p_val))
                except Exception as e:
                    logger.warning(f"Mann-Whitney test failed for {feature_name}: {e}")
        
        # Paired tests (for matched samples)
        if len(orig_features.get('pitches', [])) > 0 and len(recon_features.get('pitches', [])) > 0:
            min_len = min(len(orig_features['pitches']), len(recon_features['pitches']))
            
            if min_len > 1:
                # Wilcoxon signed-rank test
                try:
                    w_stat, p_val = stats.wilcoxon(
                        orig_features['pitches'][:min_len],
                        recon_features['pitches'][:min_len]
                    )
                    tests.wilcoxon_signed_rank = (w_stat, p_val)
                except Exception as e:
                    logger.warning(f"Wilcoxon test failed: {e}")
                
                # Paired t-test
                try:
                    t_stat, p_val = stats.ttest_rel(
                        orig_features['pitches'][:min_len],
                        recon_features['pitches'][:min_len]
                    )
                    tests.paired_t_test = (t_stat, p_val)
                except Exception as e:
                    logger.warning(f"Paired t-test failed: {e}")
        
        # F-test for variance
        if len(orig_features.get('pitches', [])) > 1 and len(recon_features.get('pitches', [])) > 1:
            var_orig = np.var(orig_features['pitches'])
            var_recon = np.var(recon_features['pitches'])
            
            if var_orig > 0 and var_recon > 0:
                f_stat = var_orig / var_recon
                df1 = len(orig_features['pitches']) - 1
                df2 = len(recon_features['pitches']) - 1
                p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
                tests.f_test_variance = (f_stat, p_val)
        
        # Determine if differences are significant
        p_values = [
            tests.mann_whitney_pitch[1],
            tests.mann_whitney_velocity[1],
            tests.mann_whitney_duration[1],
            tests.wilcoxon_signed_rank[1] if tests.wilcoxon_signed_rank[1] > 0 else 1.0,
            tests.paired_t_test[1] if tests.paired_t_test[1] > 0 else 1.0
        ]
        
        significant_tests = sum(1 for p in p_values if p < self.alpha and p > 0)
        tests.significant_difference = significant_tests > len(p_values) / 2
        
        return tests
    
    def _analyze_outliers(
        self,
        orig_features: Dict[str, np.ndarray],
        recon_features: Dict[str, np.ndarray]
    ) -> OutlierAnalysis:
        """
        Detect and analyze outliers in both datasets.
        
        Args:
            orig_features: Original features
            recon_features: Reconstructed features
            
        Returns:
            OutlierAnalysis with results
        """
        analysis = OutlierAnalysis()
        
        # Analyze pitch outliers
        orig_pitches = orig_features.get('pitches', np.array([]))
        recon_pitches = recon_features.get('pitches', np.array([]))
        
        if len(orig_pitches) > 0:
            # Z-score method
            z_scores = np.abs(stats.zscore(orig_pitches))
            analysis.zscore_outliers_original = np.sum(z_scores > 3)
            
            # IQR method
            q1 = np.percentile(orig_pitches, 25)
            q3 = np.percentile(orig_pitches, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = (orig_pitches < lower_bound) | (orig_pitches > upper_bound)
            analysis.iqr_outliers_original = np.sum(iqr_outliers)
            analysis.outlier_notes_original = list(np.where(iqr_outliers)[0])
            
            if len(orig_pitches) > 0:
                analysis.outlier_ratio_original = len(analysis.outlier_notes_original) / len(orig_pitches)
        
        if len(recon_pitches) > 0:
            # Z-score method
            z_scores = np.abs(stats.zscore(recon_pitches))
            analysis.zscore_outliers_reconstructed = np.sum(z_scores > 3)
            
            # IQR method
            q1 = np.percentile(recon_pitches, 25)
            q3 = np.percentile(recon_pitches, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = (recon_pitches < lower_bound) | (recon_pitches > upper_bound)
            analysis.iqr_outliers_reconstructed = np.sum(iqr_outliers)
            analysis.outlier_notes_reconstructed = list(np.where(iqr_outliers)[0])
            
            if len(recon_pitches) > 0:
                analysis.outlier_ratio_reconstructed = len(analysis.outlier_notes_reconstructed) / len(recon_pitches)
        
        # Calculate outlier preservation rate
        if analysis.outlier_notes_original:
            # Check how many original outliers are preserved
            preserved_outliers = 0
            for idx in analysis.outlier_notes_original:
                if idx < len(recon_pitches):
                    # Check if the corresponding note in reconstructed is also an outlier
                    if idx in analysis.outlier_notes_reconstructed:
                        preserved_outliers += 1
            
            analysis.outlier_preservation_rate = preserved_outliers / len(analysis.outlier_notes_original)
        
        # Advanced outlier detection (if sklearn is available)
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.neighbors import LocalOutlierFactor
            
            if len(orig_pitches) > 10:
                # Isolation Forest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(orig_pitches.reshape(-1, 1))
                analysis.isolation_forest_anomalies = np.sum(outliers == -1)
                
                # Local Outlier Factor
                lof = LocalOutlierFactor(n_neighbors=min(20, len(orig_pitches) - 1))
                outliers = lof.fit_predict(orig_pitches.reshape(-1, 1))
                analysis.local_outlier_factor = np.mean(lof.negative_outlier_factor_)
                
        except ImportError:
            logger.warning("sklearn not available for advanced outlier detection")
        
        return analysis
    
    def _analyze_temporal_patterns(
        self,
        orig_features: Dict[str, np.ndarray],
        recon_features: Dict[str, np.ndarray]
    ) -> TemporalAnalysis:
        """
        Analyze temporal statistical patterns.
        
        Args:
            orig_features: Original features
            recon_features: Reconstructed features
            
        Returns:
            TemporalAnalysis with results
        """
        analysis = TemporalAnalysis()
        
        orig_onsets = orig_features.get('onsets', np.array([]))
        recon_onsets = recon_features.get('onsets', np.array([]))
        
        if len(orig_onsets) > 10 and len(recon_onsets) > 10:
            # Calculate autocorrelation
            analysis.autocorrelation_original = self._calculate_autocorrelation(orig_onsets)
            analysis.autocorrelation_reconstructed = self._calculate_autocorrelation(recon_onsets)
            
            # Compare autocorrelation patterns
            min_len = min(len(analysis.autocorrelation_original), 
                         len(analysis.autocorrelation_reconstructed))
            if min_len > 0:
                diff = np.mean(np.abs(
                    np.array(analysis.autocorrelation_original[:min_len]) -
                    np.array(analysis.autocorrelation_reconstructed[:min_len])
                ))
                analysis.partial_autocorrelation_diff = diff
            
            # Test for stationarity (simplified)
            analysis.stationarity_original = self._test_stationarity(orig_onsets)
            analysis.stationarity_reconstructed = self._test_stationarity(recon_onsets)
            
            # Calculate trend similarity
            analysis.trend_similarity = self._calculate_trend_similarity(
                orig_onsets, recon_onsets
            )
            
            # Dynamic time warping distance
            analysis.dynamic_time_warping_distance = self._calculate_dtw_distance(
                orig_features.get('pitches', np.array([])),
                recon_features.get('pitches', np.array([]))
            )
        
        return analysis
    
    def _calculate_js_divergence(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence between distributions."""
        # Create histograms
        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bins = np.linspace(min_val, max_val, 50)
        
        hist1, _ = np.histogram(data1, bins=bins)
        hist2, _ = np.histogram(data2, bins=bins)
        
        # Normalize to probabilities
        p = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
        q = hist2 / hist2.sum() if hist2.sum() > 0 else hist2
        
        # Calculate JS divergence
        m = (p + q) / 2
        divergence = 0.0
        
        for i in range(len(p)):
            if p[i] > 0 and m[i] > 0:
                divergence += p[i] * np.log(p[i] / m[i])
            if q[i] > 0 and m[i] > 0:
                divergence += q[i] * np.log(q[i] / m[i])
        
        return divergence / 2
    
    def _calculate_hellinger_distance(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Hellinger distance between distributions."""
        # Create histograms
        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bins = np.linspace(min_val, max_val, 50)
        
        hist1, _ = np.histogram(data1, bins=bins)
        hist2, _ = np.histogram(data2, bins=bins)
        
        # Normalize
        p = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
        q = hist2 / hist2.sum() if hist2.sum() > 0 else hist2
        
        # Calculate Hellinger distance
        distance = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)
        
        return distance
    
    def _calculate_tv_distance(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Total Variation distance between distributions."""
        # Create histograms
        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bins = np.linspace(min_val, max_val, 50)
        
        hist1, _ = np.histogram(data1, bins=bins)
        hist2, _ = np.histogram(data2, bins=bins)
        
        # Normalize
        p = hist1 / hist1.sum() if hist1.sum() > 0 else hist1
        q = hist2 / hist2.sum() if hist2.sum() > 0 else hist2
        
        # Calculate TV distance
        distance = np.sum(np.abs(p - q)) / 2
        
        return distance
    
    def _perform_chi_square_test(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """Perform chi-square test on categorical data."""
        # Create contingency table
        unique_values = np.unique(np.concatenate([data1, data2]))
        
        if len(unique_values) > 100:  # Too many categories
            # Bin the data
            bins = np.linspace(unique_values.min(), unique_values.max(), 20)
            data1_binned = np.digitize(data1, bins)
            data2_binned = np.digitize(data2, bins)
            unique_values = np.unique(np.concatenate([data1_binned, data2_binned]))
            data1 = data1_binned
            data2 = data2_binned
        
        # Count occurrences
        counts1 = np.array([np.sum(data1 == val) for val in unique_values])
        counts2 = np.array([np.sum(data2 == val) for val in unique_values])
        
        # Create contingency table
        contingency = np.array([counts1, counts2])
        
        # Perform chi-square test
        try:
            chi2_stat, p_val, _, _ = chi2_contingency(contingency)
            return chi2_stat, p_val
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return 0.0, 1.0
    
    def _calculate_distribution_similarity(self, comparison: DistributionComparison) -> float:
        """
        Calculate overall distribution similarity score.
        
        Args:
            comparison: Distribution comparison results
            
        Returns:
            Similarity score (0-1)
        """
        scores = []
        
        # KS test score (p-value indicates similarity)
        if comparison.ks_pvalue > 0:
            scores.append(min(1.0, comparison.ks_pvalue * 10))  # Scale p-value
        
        # Wasserstein distance (lower is better)
        if comparison.wasserstein_distance >= 0:
            # Normalize (assume max reasonable distance is 100)
            scores.append(max(0.0, 1.0 - comparison.wasserstein_distance / 100))
        
        # JS divergence (lower is better)
        if comparison.jensen_shannon_divergence >= 0:
            scores.append(max(0.0, 1.0 - comparison.jensen_shannon_divergence))
        
        # Hellinger distance (lower is better, already 0-1)
        if comparison.hellinger_distance >= 0:
            scores.append(1.0 - comparison.hellinger_distance)
        
        # Chi-square p-value
        if comparison.chi_square_pvalue > 0:
            scores.append(min(1.0, comparison.chi_square_pvalue * 10))
        
        return statistics.mean(scores) if scores else 0.5
    
    def _calculate_mutual_information(
        self,
        orig_features: Dict[str, np.ndarray],
        recon_features: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate mutual information between feature sets.
        
        Args:
            orig_features: Original features
            recon_features: Reconstructed features
            
        Returns:
            Mutual information score
        """
        try:
            from sklearn.metrics import mutual_info_score
            
            orig_pitches = orig_features.get('pitches', np.array([]))
            recon_pitches = recon_features.get('pitches', np.array([]))
            
            if len(orig_pitches) > 0 and len(recon_pitches) > 0:
                min_len = min(len(orig_pitches), len(recon_pitches))
                
                # Discretize for mutual information
                orig_discrete = np.digitize(orig_pitches[:min_len], bins=np.arange(0, 128, 5))
                recon_discrete = np.digitize(recon_pitches[:min_len], bins=np.arange(0, 128, 5))
                
                mi = mutual_info_score(orig_discrete, recon_discrete)
                
                # Normalize by entropy
                h_orig = stats.entropy(np.bincount(orig_discrete))
                h_recon = stats.entropy(np.bincount(recon_discrete))
                max_entropy = max(h_orig, h_recon)
                
                if max_entropy > 0:
                    return mi / max_entropy
                
        except ImportError:
            logger.warning("sklearn not available for mutual information calculation")
        except Exception as e:
            logger.warning(f"Mutual information calculation failed: {e}")
        
        return 0.0
    
    def _calculate_cross_correlation(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray
    ) -> Tuple[float, int]:
        """
        Calculate cross-correlation and find peak.
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            Tuple of (peak correlation, lag at peak)
        """
        if len(signal1) < 10 or len(signal2) < 10:
            return 0.0, 0
        
        # Normalize signals
        signal1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
        signal2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)
        
        # Calculate cross-correlation
        correlation = np.correlate(signal1, signal2, mode='same')
        
        # Normalize by length
        correlation = correlation / len(signal1)
        
        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_value = correlation[peak_idx]
        lag = peak_idx - len(correlation) // 2
        
        return float(peak_value), int(lag)
    
    def _calculate_autocorrelation(self, signal: np.ndarray, max_lag: int = 50) -> List[float]:
        """
        Calculate autocorrelation function.
        
        Args:
            signal: Input signal
            max_lag: Maximum lag to compute
            
        Returns:
            List of autocorrelation values
        """
        if len(signal) < 10:
            return []
        
        # Normalize signal
        signal = signal - np.mean(signal)
        
        autocorr = []
        for lag in range(min(max_lag, len(signal) // 2)):
            if lag == 0:
                autocorr.append(1.0)
            else:
                c = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
                autocorr.append(c if not np.isnan(c) else 0.0)
        
        return autocorr
    
    def _test_stationarity(self, signal: np.ndarray) -> bool:
        """
        Test if time series is stationary (simplified).
        
        Args:
            signal: Time series signal
            
        Returns:
            Whether signal appears stationary
        """
        if len(signal) < 20:
            return True  # Too short to test
        
        # Split into two halves
        mid = len(signal) // 2
        first_half = signal[:mid]
        second_half = signal[mid:]
        
        # Compare statistics
        mean_diff = abs(np.mean(first_half) - np.mean(second_half))
        std_diff = abs(np.std(first_half) - np.std(second_half))
        
        # Simple stationarity check
        overall_mean = np.mean(signal)
        overall_std = np.std(signal)
        
        if overall_mean > 0:
            mean_stationary = mean_diff / overall_mean < 0.2
        else:
            mean_stationary = mean_diff < 1.0
        
        if overall_std > 0:
            std_stationary = std_diff / overall_std < 0.2
        else:
            std_stationary = std_diff < 1.0
        
        return mean_stationary and std_stationary
    
    def _calculate_trend_similarity(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray
    ) -> float:
        """
        Calculate similarity of trends in two signals.
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            Trend similarity score (0-1)
        """
        if len(signal1) < 10 or len(signal2) < 10:
            return 0.5
        
        # Fit linear trends
        x1 = np.arange(len(signal1))
        x2 = np.arange(len(signal2))
        
        try:
            # Linear regression for trend
            slope1, intercept1 = np.polyfit(x1, signal1, 1)
            slope2, intercept2 = np.polyfit(x2, signal2, 1)
            
            # Compare slopes (normalized)
            max_slope = max(abs(slope1), abs(slope2))
            if max_slope > 0:
                slope_similarity = 1.0 - abs(slope1 - slope2) / (2 * max_slope)
            else:
                slope_similarity = 1.0
            
            return max(0.0, min(1.0, slope_similarity))
            
        except Exception as e:
            logger.warning(f"Trend calculation failed: {e}")
            return 0.5
    
    def _calculate_dtw_distance(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """
        Calculate Dynamic Time Warping distance (simplified).
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            DTW distance
        """
        if len(signal1) == 0 or len(signal2) == 0:
            return 0.0
        
        # Limit signal length for computational efficiency
        max_len = 100
        if len(signal1) > max_len:
            signal1 = signal1[:max_len]
        if len(signal2) > max_len:
            signal2 = signal2[:max_len]
        
        n, m = len(signal1), len(signal2)
        
        # Create distance matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(signal1[i-1] - signal2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]    # match
                )
        
        # Normalize by path length
        distance = dtw_matrix[n, m] / (n + m)
        
        return distance
    
    def _calculate_overall_similarity(self, metrics: StatisticalMetrics) -> float:
        """
        Calculate overall statistical similarity score.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            Overall similarity score (0-1)
        """
        scores = []
        weights = []
        
        # Distribution similarity scores
        for dist_comp in metrics.distribution_comparison.values():
            scores.append(dist_comp.distribution_similarity_score)
            weights.append(0.2)
        
        # Correlation scores
        correlation_score = (
            abs(metrics.correlation_analysis.pitch_correlation) * 0.4 +
            abs(metrics.correlation_analysis.velocity_correlation) * 0.2 +
            abs(metrics.correlation_analysis.duration_correlation) * 0.2 +
            metrics.correlation_analysis.mutual_information * 0.2
        )
        scores.append(correlation_score)
        weights.append(0.3)
        
        # Significance test score
        if not metrics.significance_tests.significant_difference:
            scores.append(1.0)  # No significant difference is good
        else:
            # Calculate score based on p-values
            p_values = [
                metrics.significance_tests.mann_whitney_pitch[1],
                metrics.significance_tests.mann_whitney_velocity[1],
                metrics.significance_tests.mann_whitney_duration[1]
            ]
            avg_pvalue = statistics.mean([p for p in p_values if p > 0])
            scores.append(min(1.0, avg_pvalue * 10))
        weights.append(0.2)
        
        # Outlier similarity score
        outlier_diff = abs(metrics.outlier_analysis.outlier_ratio_original - 
                          metrics.outlier_analysis.outlier_ratio_reconstructed)
        outlier_score = max(0.0, 1.0 - outlier_diff * 5)  # Scale difference
        scores.append(outlier_score)
        weights.append(0.15)
        
        # Temporal similarity score
        if metrics.temporal_analysis.stationarity_original == metrics.temporal_analysis.stationarity_reconstructed:
            temporal_score = 1.0
        else:
            temporal_score = 0.5
        temporal_score = temporal_score * 0.5 + metrics.temporal_analysis.trend_similarity * 0.5
        scores.append(temporal_score)
        weights.append(0.15)
        
        # Calculate weighted average
        if sum(weights) > 0:
            overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            overall = 0.5
        
        return max(0.0, min(1.0, overall))
    
    def _calculate_statistical_scores(self, metrics: StatisticalMetrics) -> Dict[str, float]:
        """
        Calculate individual statistical scores.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            Dictionary of statistical scores
        """
        scores = {}
        
        # Distribution scores
        for name, comp in metrics.distribution_comparison.items():
            scores[f'{name}_distribution_similarity'] = comp.distribution_similarity_score
            scores[f'{name}_ks_test_passed'] = 1.0 if comp.distributions_identical else 0.0
        
        # Correlation scores
        scores['pitch_correlation'] = abs(metrics.correlation_analysis.pitch_correlation)
        scores['velocity_correlation'] = abs(metrics.correlation_analysis.velocity_correlation)
        scores['duration_correlation'] = abs(metrics.correlation_analysis.duration_correlation)
        scores['mutual_information'] = metrics.correlation_analysis.mutual_information
        
        # Significance scores
        scores['statistical_significance'] = 0.0 if metrics.significance_tests.significant_difference else 1.0
        
        # Outlier scores
        scores['outlier_consistency'] = max(0.0, 1.0 - abs(
            metrics.outlier_analysis.outlier_ratio_original - 
            metrics.outlier_analysis.outlier_ratio_reconstructed
        ) * 10)
        scores['outlier_preservation'] = metrics.outlier_analysis.outlier_preservation_rate
        
        # Temporal scores
        scores['stationarity_preservation'] = (1.0 if 
            metrics.temporal_analysis.stationarity_original == 
            metrics.temporal_analysis.stationarity_reconstructed else 0.0)
        scores['trend_similarity'] = metrics.temporal_analysis.trend_similarity
        
        return scores
    
    def _generate_statistical_warnings(self, metrics: StatisticalMetrics) -> List[str]:
        """
        Generate warnings about statistical issues.
        
        Args:
            metrics: Collected metrics
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Distribution warnings
        for name, comp in metrics.distribution_comparison.items():
            if comp.ks_pvalue < 0.01:
                warnings.append(f"{name} distribution significantly different (KS p-value: {comp.ks_pvalue:.4f})")
            
            if comp.wasserstein_distance > 50:
                warnings.append(f"Large Wasserstein distance for {name}: {comp.wasserstein_distance:.2f}")
        
        # Correlation warnings
        if abs(metrics.correlation_analysis.pitch_correlation) < 0.5:
            warnings.append(f"Low pitch correlation: {metrics.correlation_analysis.pitch_correlation:.3f}")
        
        if abs(metrics.correlation_analysis.velocity_correlation) < 0.3:
            warnings.append(f"Low velocity correlation: {metrics.correlation_analysis.velocity_correlation:.3f}")
        
        if metrics.correlation_analysis.mutual_information < 0.3:
            warnings.append(f"Low mutual information: {metrics.correlation_analysis.mutual_information:.3f}")
        
        # Significance warnings
        if metrics.significance_tests.significant_difference:
            warnings.append("Statistically significant differences detected between original and reconstructed")
        
        # Outlier warnings
        outlier_diff = abs(metrics.outlier_analysis.outlier_ratio_original - 
                          metrics.outlier_analysis.outlier_ratio_reconstructed)
        if outlier_diff > 0.1:
            warnings.append(f"Outlier ratio mismatch: {outlier_diff:.2%}")
        
        if metrics.outlier_analysis.outlier_preservation_rate < 0.5:
            warnings.append(f"Poor outlier preservation: {metrics.outlier_analysis.outlier_preservation_rate:.1%}")
        
        # Temporal warnings
        if metrics.temporal_analysis.stationarity_original != metrics.temporal_analysis.stationarity_reconstructed:
            warnings.append("Stationarity properties not preserved")
        
        if metrics.temporal_analysis.trend_similarity < 0.7:
            warnings.append(f"Different trends detected (similarity: {metrics.temporal_analysis.trend_similarity:.2f})")
        
        if metrics.temporal_analysis.dynamic_time_warping_distance > 10:
            warnings.append(f"High DTW distance: {metrics.temporal_analysis.dynamic_time_warping_distance:.2f}")
        
        return warnings