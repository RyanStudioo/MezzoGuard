from mezzoguard import CONTENTGUARD
from mezzoguard.content_guard import Guard

with Guard(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_SMALL) as model:
    print(model.pipeline)
    scan = model.scan("Hello world")
    print(scan.scores)

print(model.pipeline)