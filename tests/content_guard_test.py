from mezzoguard import ContentGuardModel

model = ContentGuardModel(name=r"C:\Users\yanya\PycharmProjects\ContentGuard\models\v1-models\mezzo-content-guard-large\checkpoint-25506")

while True:
    user_input = input("Enter a prompt (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print(model.scan(user_input))