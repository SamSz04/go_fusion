"""
Metrics computation and comparison for GO fusion evaluation.

Provides utilities to:
- Compute speedup ratios against baselines
- Compute memory reduction metrics
- Analyze fusion cluster statistics
- Format results into a readable comparison table
"""

from typing import Dict, List, Any, Optional


class FusionMetrics:
    """Metrics for evaluating GO fusion policy against baselines."""

    @staticmethod
    def compute_speedup(go_runtime: float, baseline_runtime: float) -> float:
        """Compute speedup ratio of GO over a baseline.

        Speedup > 1.0 means GO is faster.
        Speedup < 1.0 means baseline is faster.

        Args:
            go_runtime: Runtime achieved by the GO policy.
            baseline_runtime: Runtime achieved by the baseline.

        Returns:
            Speedup ratio (baseline_runtime / go_runtime).
        """
        if go_runtime <= 0:
            return float("inf")
        return baseline_runtime / go_runtime

    @staticmethod
    def compute_speedup_percentage(go_runtime: float, baseline_runtime: float) -> float:
        """Compute percentage improvement of GO over a baseline.

        Positive means GO is faster; negative means baseline is faster.

        Args:
            go_runtime: Runtime achieved by the GO policy.
            baseline_runtime: Runtime achieved by the baseline.

        Returns:
            Percentage improvement ((baseline - go) / baseline * 100).
        """
        if baseline_runtime <= 0:
            return 0.0
        return (baseline_runtime - go_runtime) / baseline_runtime * 100.0

    @staticmethod
    def compute_memory_reduction(
        go_bytes: float, baseline_bytes: float
    ) -> float:
        """Compute memory traffic reduction of GO over a baseline.

        Positive means GO uses less memory bandwidth.

        Args:
            go_bytes: Total memory bytes accessed under GO fusion.
            baseline_bytes: Total memory bytes accessed under baseline.

        Returns:
            Percentage reduction ((baseline - go) / baseline * 100).
        """
        if baseline_bytes <= 0:
            return 0.0
        return (baseline_bytes - go_bytes) / baseline_bytes * 100.0

    @staticmethod
    def compute_cluster_stats(clusters: List[List[Any]]) -> Dict[str, Any]:
        """Compute statistics about fusion clusters.

        Args:
            clusters: List of clusters, where each cluster is a list of
                      instructions/node indices.

        Returns:
            Dictionary with cluster statistics:
            - num_clusters: Total number of fusion clusters.
            - avg_size: Average number of ops per cluster.
            - max_size: Size of the largest cluster.
            - min_size: Size of the smallest cluster.
            - median_size: Median cluster size.
            - single_op_clusters: Number of clusters with only one op.
            - multi_op_clusters: Number of clusters with more than one op.
            - size_distribution: Dict mapping size -> count of clusters.
        """
        if not clusters:
            return {
                "num_clusters": 0,
                "avg_size": 0.0,
                "max_size": 0,
                "min_size": 0,
                "median_size": 0.0,
                "single_op_clusters": 0,
                "multi_op_clusters": 0,
                "size_distribution": {},
            }

        sizes = [len(c) for c in clusters]
        sizes_sorted = sorted(sizes)
        n = len(sizes_sorted)

        if n % 2 == 0:
            median = (sizes_sorted[n // 2 - 1] + sizes_sorted[n // 2]) / 2.0
        else:
            median = float(sizes_sorted[n // 2])

        # Size distribution
        size_dist: Dict[int, int] = {}
        for s in sizes:
            size_dist[s] = size_dist.get(s, 0) + 1

        return {
            "num_clusters": len(clusters),
            "avg_size": sum(sizes) / len(sizes),
            "max_size": max(sizes),
            "min_size": min(sizes),
            "median_size": median,
            "single_op_clusters": sum(1 for s in sizes if s == 1),
            "multi_op_clusters": sum(1 for s in sizes if s > 1),
            "size_distribution": dict(sorted(size_dist.items())),
        }

    @staticmethod
    def compare_all(
        go_runtime: float, baselines: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Compare GO against all baselines.

        Args:
            go_runtime: Runtime achieved by GO policy.
            baselines: Dictionary mapping baseline name -> runtime.

        Returns:
            Dictionary mapping baseline name -> {
                'baseline_runtime': float,
                'go_runtime': float,
                'speedup': float,
                'improvement_pct': float,
            }
        """
        results = {}
        for name, baseline_runtime in baselines.items():
            results[name] = {
                "baseline_runtime": baseline_runtime,
                "go_runtime": go_runtime,
                "speedup": FusionMetrics.compute_speedup(go_runtime, baseline_runtime),
                "improvement_pct": FusionMetrics.compute_speedup_percentage(
                    go_runtime, baseline_runtime
                ),
            }
        return results

    @staticmethod
    def format_results_table(
        go_runtime: float,
        baselines: Dict[str, float],
        go_cluster_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format comparison results as a readable table.

        Args:
            go_runtime: Runtime achieved by GO policy.
            baselines: Dictionary mapping baseline name -> runtime.
            go_cluster_stats: Optional cluster statistics from GO.

        Returns:
            Formatted string with the comparison table.
        """
        comparisons = FusionMetrics.compare_all(go_runtime, baselines)

        # Header
        lines = []
        lines.append("")
        lines.append("=" * 80)
        lines.append("  GO Fusion Evaluation Results")
        lines.append("=" * 80)
        lines.append("")

        # Runtime comparison table
        lines.append(
            f"  {'Method':<25s} {'Runtime (s)':<15s} {'Speedup':<12s} {'Improvement':<12s}"
        )
        lines.append("  " + "-" * 64)

        # GO row first
        lines.append(
            f"  {'GO (learned)':.<25s} {go_runtime:<15.6e} {'---':.<12s} {'---':.<12s}"
        )

        # Baseline rows sorted by runtime (fastest first)
        sorted_baselines = sorted(comparisons.items(), key=lambda x: x[1]["baseline_runtime"])
        for name, comp in sorted_baselines:
            display_name = name.replace("_", " ").title()
            speedup_str = f"{comp['speedup']:.3f}x"
            if comp["improvement_pct"] >= 0:
                improvement_str = f"+{comp['improvement_pct']:.1f}%"
            else:
                improvement_str = f"{comp['improvement_pct']:.1f}%"
            lines.append(
                f"  {display_name:<25s} {comp['baseline_runtime']:<15.6e} "
                f"{speedup_str:<12s} {improvement_str:<12s}"
            )

        lines.append("")

        # Summary
        best_baseline_name = min(baselines, key=baselines.get)
        best_baseline_runtime = baselines[best_baseline_name]
        vs_best = FusionMetrics.compute_speedup_percentage(go_runtime, best_baseline_runtime)

        lines.append(f"  Best baseline: {best_baseline_name.replace('_', ' ').title()} "
                      f"({best_baseline_runtime:.6e} s)")
        if vs_best >= 0:
            lines.append(f"  GO improvement over best baseline: +{vs_best:.1f}%")
        else:
            lines.append(f"  GO vs best baseline: {vs_best:.1f}% (baseline is faster)")

        no_fusion_runtime = baselines.get("no_fusion")
        if no_fusion_runtime is not None:
            vs_nofusion = FusionMetrics.compute_speedup_percentage(
                go_runtime, no_fusion_runtime
            )
            lines.append(f"  GO improvement over no fusion: +{vs_nofusion:.1f}%")

        # Cluster stats
        if go_cluster_stats:
            lines.append("")
            lines.append("  Cluster Statistics:")
            lines.append(f"    Total clusters: {go_cluster_stats['num_clusters']}")
            lines.append(f"    Average size:   {go_cluster_stats['avg_size']:.1f}")
            lines.append(f"    Max size:       {go_cluster_stats['max_size']}")
            lines.append(f"    Min size:       {go_cluster_stats['min_size']}")
            lines.append(f"    Median size:    {go_cluster_stats['median_size']:.1f}")
            lines.append(
                f"    Single-op:      {go_cluster_stats['single_op_clusters']} "
                f"({go_cluster_stats['single_op_clusters'] / max(go_cluster_stats['num_clusters'], 1) * 100:.0f}%)"
            )
            lines.append(
                f"    Multi-op:       {go_cluster_stats['multi_op_clusters']} "
                f"({go_cluster_stats['multi_op_clusters'] / max(go_cluster_stats['num_clusters'], 1) * 100:.0f}%)"
            )

        lines.append("")
        lines.append("=" * 80)
        lines.append("")

        return "\n".join(lines)
