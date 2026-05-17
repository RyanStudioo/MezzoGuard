from mezzoguard import CONTENTGUARD
from mezzoguard.modules.content_guard import Guard

with Guard(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE_PREVIEW) as model:
    print(model.pipeline)
    model.scan("Hello world")

print(model.pipeline)