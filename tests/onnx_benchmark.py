import time
import statistics

from mezzoguard import CONTENTGUARD
from mezzoguard.content_guard import Guard

MESSAGES = [
    "Hello, how are you today?",
    "I need help with my order",
    "Can you recommend a good restaurant?",
    "The weather is nice outside",
    "What time is the meeting?",
    "I love this product, it's amazing!",
    "This is terrible, I want a refund",
    "Can I speak to a manager?",
    "Thank you for your help",
    "I have a question about pricing",
]

NUM_RUNS = 10
WARMUP_RUNS = 3


def benchmark(model: Guard, label: str) -> dict:
    for _ in range(WARMUP_RUNS):
        for msg in MESSAGES:
            model.scan(msg)

    latencies = []
    for _ in range(NUM_RUNS):
        for msg in MESSAGES:
            start = time.perf_counter()
            model.scan(msg)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    return {
        "label": label,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "total_calls": len(latencies),
    }


def print_result(result: dict) -> None:
    print(f"\n--- {result['label']} ---")
    print(f"  Mean:     {result['mean_ms']:.2f} ms")
    print(f"  Median:   {result['median_ms']:.2f} ms")
    print(f"  Std Dev:  {result['stdev_ms']:.2f} ms")
    print(f"  Min:      {result['min_ms']:.2f} ms")
    print(f"  Max:      {result['max_ms']:.2f} ms")
    print(f"  Calls:    {result['total_calls']}")


if __name__ == "__main__":
    model_name = CONTENTGUARD.MEZZO_CONTENT_GUARD_SMALL

    print(f"Model: {model_name}")
    print(f"Messages: {len(MESSAGES)}")
    print(f"Runs per message: {NUM_RUNS}")
    print(f"Warmup runs: {WARMUP_RUNS}")

    with Guard(name=model_name) as guard:
        standard = benchmark(guard, "PyTorch (standard)")
        print_result(standard)

    with Guard(name=model_name, use_onnx=True) as guard:
        onnx = benchmark(guard, "ONNX")
        print_result(onnx)

    speedup = standard["mean_ms"] / onnx["mean_ms"]
    print(f"\n=== Summary ===")
    print(f"PyTorch mean: {standard['mean_ms']:.2f} ms")
    print(f"ONNX mean:    {onnx['mean_ms']:.2f} ms")
    print(f"Speedup:      {speedup:.2f}x")
