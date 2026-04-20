# 调用惯例，位置参数
METH_VARARGS = 1

# 调用惯例，关键字参数
METH_KEYWORDS = 2

# 调用惯例，不接受参数，C层第二个参数接受NULL
METH_NOARGS = 4

# 调用惯例，只接收一个参数，C层第二个参数接受Python对象
METH_O = 8

# 调用惯例
METH_FASTCALL = 128

# 调用惯例
METH_METHOD = 512

# 绑定惯例，@classmethod，C层第一个参数接受类对象
METH_CLASS = 16

# 绑定惯例，@staticmethod，C层第一个参数接受NULL
METH_STATIC = 32


METH_COEXIST = 64