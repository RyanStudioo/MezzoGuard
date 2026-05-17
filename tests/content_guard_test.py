from mezzoguard import Guard, CONTENTGUARD
from mezzoguard.modules.content_guard.categories import ContentGuardCheck, Category

model = Guard(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE_PREVIEW, categories=[
    ContentGuardCheck(Category.HATE_SPEECH, threshold=0.5)
])

while True:
    user_input = input("Enter a prompt (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    scanned = model.scan(user_input)
    print(scanned.violations)