def fn():
    print("hello")
    print("fn")

def fn1(a=10, b=20):
    print(a, b)

def fn2(*args):
    print(args)

def fn3(**kwargs):
    print(kwargs)

def fn4(*args, **kwargs):
    print(args)
    print(kwargs)


fn1(b=30)

t = 1, 2, 3, 4, 5
d = {"name": "hojeong"}

fn2(t)

fn3(d)
fn3(name="hojeong")

fn4(t, d)
fn4(1, 2, 3, name="hojeong")
