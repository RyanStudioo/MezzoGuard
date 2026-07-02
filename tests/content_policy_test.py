from mezzoguard import CONTENTGUARD
from mezzoguard.content_guard import ContentPolicy, BaseCategory, Guard

model = Guard(CONTENTGUARD.MEZZO_CONTENT_GUARD_SMALL)
content_policy = ContentPolicy().add_threshold(BaseCategory.SEXUAL, 0.5)

sexual_query = "I want to fuck you"
benign_query = "I want to have a nice day"
violent_query = "I want to kill you"

result_1 = model.scan(text=sexual_query)
print(content_policy.evaluate(result_1))
# True

result_2 = model.scan(text=benign_query)
print(content_policy.evaluate(result_2))
# False

result_3 = model.scan(text=violent_query)
print(content_policy.evaluate(result_3))
# False