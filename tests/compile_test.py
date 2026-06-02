import time

from mezzoguard import CONTENTGUARD
from mezzoguard.content_guard import Guard

with Guard(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE) as guard:
	warmup = [guard.scan("Hello world") for _ in range(3)]
	start = time.time()
	guard.scan("Uncompiled model latency test")
	end = time.time()
	print(f"Uncompiled Latency: {(end - start) * 1000} ms")

with Guard(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE, torch_compile=True) as guard:
	warmup = [guard.scan("Hello world") for _ in range(3)]
	start = time.time()
	guard.scan("Compiled model latency test")
	end = time.time()
	print(f"Compiled latency: {(end - start) * 1000} ms")
