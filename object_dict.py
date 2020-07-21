# wrapper to make a dict look like a class, to simplify access to members
# https://goodcode.io/articles/python-dict-object/
    
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        
class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
            
#test
'''
d = {'a': 1, 'b': 2}

o1 = objectview(d)
print(o1.a)

o2= objdict(d)
print(o2.b)
o2.c = 3
print(o2.c)

params = objdict({})
params.mixing   = 1
params.mixing_s = 1
params.phi      = 1
params.q        = 1
params.gamma    = 1/5
params.lag      = 1

print(params)
'''
