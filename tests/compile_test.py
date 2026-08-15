import time

from mezzoguard import CONTENTGUARD
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
    uncompiled = benchmark(CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE)
    print(f"Uncompiled Latency: {uncompiled} ms")

    compiled = benchmark(CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE, torch_compile=True)
    print(f"Compiled Latency: {compiled} ms")
