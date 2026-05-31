"""Multi-backend benchmarking for wafer defect detection.

Compares inference performance across PyTorch-CUDA, PyTorch-ROCm,
MIGraphX, and ONNX Runtime backends.
"""

import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    backend: str
    model_name: str
    batch_size: int
    img_size: int
    iterations: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    fps: float
    gpu_name: str = ""

    @property
    def summary(self) -> str:
        return (
            f"{self.backend} | {self.model_name} | "
            f"batch={self.batch_size} | "
            f"mean={self.mean_ms:.1f}ms | fps={self.fps:.0f}"
        )


class MultiBackendBenchmark:
    """Benchmark across multiple inference backends.
    
    Example:
        >>> bench = MultiBackendBenchmark()
        >>> results = bench.run_all("weights/wafer_yolov8l.pt")
        >>> bench.print_comparison(results)
    """

    def __init__(self, img_size: int = 640, iterations: int = 100):
        self.img_size = img_size
        self.iterations = iterations

    def _generate_dummy_input(self, batch_size: int = 1) -> np.ndarray:
        """Generate dummy input tensor."""
        return np.random.randint(0, 255, (batch_size, self.img_size, self.img_size, 3), dtype=np.uint8)

    def benchmark_pytorch(
        self,
        weights: str,
        device: str = "cuda",
        batch_size: int = 1,
    ) -> BenchmarkResult:
        """Benchmark PyTorch inference."""
        from ultralytics import YOLO

        model = YOLO(weights)
        dummy = self._generate_dummy_input(batch_size)

        # Warmup
        for _ in range(5):
            model.predict(dummy[0], verbose=False)

        latencies = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            model.predict(dummy[0], verbose=False)
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(latencies)
        return BenchmarkResult(
            backend=f"PyTorch-{device.upper()}",
            model_name=Path(weights).stem,
            batch_size=batch_size,
            img_size=self.img_size,
            iterations=self.iterations,
            mean_ms=mean_ms,
            p50_ms=np.percentile(latencies, 50),
            p95_ms=np.percentile(latencies, 95),
            p99_ms=np.percentile(latencies, 99),
            min_ms=np.min(latencies),
            max_ms=np.max(latencies),
            fps=1000 / mean_ms,
        )

    def benchmark_migraphx(self, model_path: str, batch_size: int = 1) -> BenchmarkResult:
        """Benchmark MIGraphX inference."""
        from .migraphx_engine import MIGraphXEngine

        engine = MIGraphXEngine(model_path)
        stats = engine.benchmark(self.iterations)

        mean_ms = stats["mean_ms"]
        return BenchmarkResult(
            backend="MIGraphX",
            model_name=Path(model_path).stem,
            batch_size=batch_size,
            img_size=self.img_size,
            iterations=self.iterations,
            mean_ms=mean_ms,
            p50_ms=stats["p50_ms"],
            p95_ms=stats["p95_ms"],
            p99_ms=stats["p99_ms"],
            min_ms=stats["min_ms"],
            max_ms=stats["max_ms"],
            fps=1000 / mean_ms,
        )

    def benchmark_onnxruntime(self, model_path: str, batch_size: int = 1) -> BenchmarkResult:
        """Benchmark ONNX Runtime inference."""
        import onnxruntime as ort

        sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        dummy = self._generate_dummy_input(batch_size)
        dummy = np.transpose(dummy, (0, 3, 1, 2)).astype(np.float32) / 255.0

        # Warmup
        for _ in range(5):
            sess.run(None, {input_name: dummy})

        latencies = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            sess.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000)

        mean_ms = np.mean(latencies)
        return BenchmarkResult(
            backend="ONNXRuntime",
            model_name=Path(model_path).stem,
            batch_size=batch_size,
            img_size=self.img_size,
            iterations=self.iterations,
            mean_ms=mean_ms,
            p50_ms=np.percentile(latencies, 50),
            p95_ms=np.percentile(latencies, 95),
            p99_ms=np.percentile(latencies, 99),
            min_ms=np.min(latencies),
            max_ms=np.max(latencies),
            fps=1000 / mean_ms,
        )

    def print_comparison(self, results: list[BenchmarkResult]) -> None:
        """Print formatted comparison table."""
        print("\n" + "=" * 80)
        print(f"{'Backend':<20} {'Model':<25} {'Mean (ms)':<12} {'FPS':<10} {'P95 (ms)':<10}")
        print("-" * 80)
        for r in sorted(results, key=lambda x: x.mean_ms):
            print(f"{r.backend:<20} {r.model_name:<25} {r.mean_ms:<12.1f} {r.fps:<10.0f} {r.p95_ms:<10.1f}")
        print("=" * 80)

    def export_json(self, results: list[BenchmarkResult], output_path: str) -> None:
        """Export results to JSON."""
        data = {
            "config": {"img_size": self.img_size, "iterations": self.iterations},
            "results": [asdict(r) for r in results],
        }
        Path(output_path).write_text(json.dumps(data, indent=2))
        logger.info(f"Results exported to {output_path}")
