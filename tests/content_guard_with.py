from mezzoguard import Guard, CONTENTGUARD

with Guard(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE_PREVIEW) as model:
    print(model.pipeline)
    model.scan("Hello world")

print(model.pipeline)