import time

from mezzoguard.content_guard import Guard


def benchmark(model_name: str, **kwargs) -> float:
    with Guard(name=model_name, **kwargs) as guard:
        for _ in range(3):
            guard.scan("Hello world")
        start = time.time()
        guard.scan("Uncompiled model latency test")
        end = time.time()
        return (end - start) * 1000


if __name__ == "__main__":
    uncompiled = benchmark("RyanStudio/Mezzo-Content-Guard-Large")
    print(f"Uncompiled Latency: {uncompiled} ms")

    compiled = benchmark("RyanStudio/Mezzo-Content-Guard-Large", torch_compile=True)
    print(f"Compiled Latency: {compiled} ms")
