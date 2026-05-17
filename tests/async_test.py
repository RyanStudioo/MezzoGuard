import asyncio
import time

from mezzoguard import CONTENTGUARD
from mezzoguard.modules.content_guard import Guard


async def main():
    model = Guard(CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE_PREVIEW)

    text = "This is a test message to check if the async scanning works correctly."

    print("=== Testing async_scan ===")
    result = await model.async_scan(text)
    print(f"Safe: {result.is_safe()}, Violations: {result.violations}")

    print("\n=== Testing async_redact ===")
    redacted = await model.async_redact(text)
    print(f"Redacted: {redacted[:80]}...")

    print("\n=== Testing concurrent execution ===")
    texts = [
        "I love everyone and everything.",
        "You are all worthless and terrible.",
        "The weather is nice today.",
        "I want to hurt myself.",
    ]
    t0 = time.time()
    results = await asyncio.gather(*[model.async_scan(t) for t in texts])
    t1 = time.time()
    for t, r in zip(texts, results):
        print(f"  {t[:40]:<42} safe={r.is_safe()}")
    print(f"  Completed {len(texts)} scans in {t1-t0:.2f}s")

    print("\n=== Testing async_redact_before_exec decorator ===")
    @model.redact_before_exec(param="msg")
    async def async_echo(msg: str) -> str:
        return msg

    clean = await async_echo("You are a terrible person and I hate you!")
    print(f"  Redacted message: {clean[:80]}")

    print("\n=== Testing async_scan_before_exec decorator ===")
    @model.scan_before_exec(param="msg", confidence=0.5)
    async def guarded_echo(msg: str) -> str:
        return msg

    try:
        await guarded_echo("I am going to kill myself!")
        print("  ERROR: should have raised UnsafePromptError")
    except Exception as e:
        print(f"  Correctly raised: {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(main())
