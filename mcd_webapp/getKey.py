def getKey1():
    with open("mcd_webapp/key1.txt") as f:
        return f.readlines()[0]

def getKey2():
    with open("mcd_webapp/key2.txt") as f:
        return f.readlines()[0]