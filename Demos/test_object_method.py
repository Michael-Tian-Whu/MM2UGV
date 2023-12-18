'''
https://piantou.blog.csdn.net/article/details/103798741
'''
class father():
    c=1#类属性
    def __init__(self) -> None:
        self.a=1#实例属性
        print(self.a)
    @classmethod
    # 只访问类属性
    # 可以用来为一个类创建一些预处理的实例
    def cm(cls):
        print(cls.c)#cls代表类本身
    @staticmethod
    # 不能访问任何属性
    def sm(b):
        print(b)


class son(father):
    def __init__(self) -> None:
        super().__init__()
        self.a=2
        print(super().a)

father.cm()


'''
https://blog.csdn.net/wanzew/article/details/106993425
'''
class A:
    pass
 
 
class B(A):
    pass
 
 
class C(A):
    def funxx(self):
        print("找到 funxx() 位于 C 中...")
 
 
class D(A):
    pass
 
 
class E(B, C):
    pass
 
 
class F(E, D):
    def funff(self):
        print("执行 F 中的 funff()...")
        super(E, self).funxx()
 
        
print(f"F 类的 MRO : {F.__mro__}")
f = F()
f.funff()

