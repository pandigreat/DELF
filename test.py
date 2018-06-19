
a = [i for i in range(10)]

def get(a):
    while True:
        for i in range(0, 10, 3):
            try:
                if i+3 < 10:
                    yield a[i:i+3]
                else :
                    raise Exception
            except:
                break;

b = get(a)
for i in range(20):
    print(next(b))

