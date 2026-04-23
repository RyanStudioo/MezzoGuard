from mezzoguard import ContentGuardModel, CONTENTGUARD

model = ContentGuardModel(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE_PREVIEW)

while True:
    user_input = input("Enter a prompt (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print(model.scan(user_input))