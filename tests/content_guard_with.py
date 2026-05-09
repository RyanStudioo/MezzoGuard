from mezzoguard import ContentGuardModel, CONTENTGUARD

with ContentGuardModel(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE_PREVIEW) as model:
    print(model.pipeline)
    model.scan("Hello world")

print(model.pipeline)