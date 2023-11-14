class Vector1:
    def __init__(self, a):
        self.value = a

    def __add__(self, a2):
        if len(self.value) != len(a2.value):
            print 'not match'
            return
        sumvalue = []
        for i in range(len(self.value)):
            sumvalue.append(self.value[i]+a2.value[i])
        return sumvalue


a = Vector1([1, 2, 3])
b = Vector1([2, 3, 4])
c = Vector1([1, 2, 3, 4])
a+b
a+c
