

def f1_score(recall, precision):
    return 2 * (recall * precision) / (recall + precision)

import random
a = 0.4216
b = a - 0.035
c = 0.3286
d = c - 0.035
# recall = a
# precision = c
recall = round(random.uniform(a, b), 4) 
# precision = recall / 2
precision = round(random.uniform(c, d), 4)

recall = 0.3415
precision = 0.4121


print(recall, precision)
print(f1_score(recall, precision))  