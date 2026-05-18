from mezzoguard import CONTENTGUARD
from mezzoguard.modules.content_guard import Guard

model = Guard(name=CONTENTGUARD.MEZZO_CONTENT_GUARD_LARGE_PREVIEW)

while True:
    user_input = input("Enter a prompt (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    scanned = model.scan(user_input)
    print(scanned.scores)
