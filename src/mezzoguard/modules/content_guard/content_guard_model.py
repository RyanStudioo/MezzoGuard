from mezzoguard.model import Model


class ContentGuardModel(Model):
    def __init__(self, name: str):
        super().__init__(name, task="text-classification")

    def scan(self, text: str, ):


