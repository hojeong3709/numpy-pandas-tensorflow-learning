# list - 순서가 있고 변경가능
l = [10, 20, 30, 40, 50]

l.append(100)
print(l)
l.insert(1, 200)
print(l)
l.remove(30)
print(l)
l.pop(-1)
print(l)

# tuple - 순서가 있고 변경불가능
t = (10, 20, 30)

# dictionary - 순서가 없고 변경가능
d = dict()
d = {"name": "hojeong.kwak", "age": 33}
print(d.keys())
print(d.values())
print(d.items())

d["height"] = 173
print(d)