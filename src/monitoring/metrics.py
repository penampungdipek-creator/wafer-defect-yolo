"""Prometheus metrics for wafer defect detection API."""

try:
    from prometheus_client import Counter, Histogram, Gauge

    INFERENCE_LATENCY = Histogram(
        "inference_latency_seconds",
        "Model inference latency",
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )

    INFERENCE_TOTAL = Counter(
        "inference_total",
        "Total number of inferences",
    )

    DEFECT_DISTRIBUTION = Counter(
        "defect_class_distribution",
        "Distribution of detected defect classes",
        ["defect_class"],
    )

    ACTIVE_REQUESTS = Gauge(
        "active_requests",
        "Number of currently active inference requests",
    )

except ImportError:
    # Fallback no-op metrics when prometheus_client not installed
    class _NoopMetric:
        def labels(self, **kw): return self
        def inc(self, n=1): pass
        def dec(self, n=1): pass
        def time(self):
            from contextlib import contextmanager
            @contextmanager
            def noop():
                yield
            return noop()

    INFERENCE_LATENCY = _NoopMetric()
    INFERENCE_TOTAL = _NoopMetric()
    DEFECT_DISTRIBUTION = _NoopMetric()
    ACTIVE_REQUESTS = _NoopMetric()
