#define PY_SSIZE_T_CLEAN
#include <Python.h>

// 简单的加法函数
static PyObject* simple_add(PyObject* self, PyObject* args) {
    int a, b;
    
    // 解析Python参数
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
        return NULL;
    }
    
    // 执行加法运算
    int result = a + b;
    
    // 返回Python整数对象
    return PyLong_FromLong(result);
}

// 简单的字符串处理函数
static PyObject* simple_greet(PyObject* self, PyObject* args) {
    const char* name;
    
    // 解析Python参数
    if (!PyArg_ParseTuple(args, "s", &name)) {
        return NULL;
    }
    
    // 创建问候语
    char greeting[100];
    snprintf(greeting, sizeof(greeting), "Hello, %s!", name);
    
    // 返回Python字符串对象
    return PyUnicode_FromString(greeting);
}

// 简单的列表求和函数
static PyObject* simple_sum_list(PyObject* self, PyObject* args) {
    PyObject* list;
    
    // 解析Python参数
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list)) {
        return NULL;
    }
    
    // 获取列表长度
    Py_ssize_t size = PyList_Size(list);
    if (size < 0) {
        return NULL;
    }
    
    // 遍历列表并求和
    long sum = 0;
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject* item = PyList_GetItem(list, i);
        if (!PyLong_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "All items must be integers");
            return NULL;
        }
        sum += PyLong_AsLong(item);
    }
    
    // 返回总和
    return PyLong_FromLong(sum);
}

// 方法定义数组
static PyMethodDef SimpleMethods[] = {
    {"add", simple_add, METH_VARARGS, "Add two integers."},
    {"greet", simple_greet, METH_VARARGS, "Return a greeting message."},
    {"sum_list", simple_sum_list, METH_VARARGS, "Sum all integers in a list."},
    {NULL, NULL, 0, NULL}  // 哨兵值
};

// 模块定义
static struct PyModuleDef simplemodule = {
    PyModuleDef_HEAD_INIT,
    "simple_extension",           // 模块名
    "A simple Python C extension example.", // 模块文档
    -1,                          // 模块状态大小
    SimpleMethods,               // 方法定义
    NULL,                        // 模块遍历函数
    NULL,                        // 清理函数
    NULL,                        // 模块支持函数
    NULL                         // 调试信息
};

// 模块初始化函数
// Python解释器执行import module时，会调用此函数
PyMODINIT_FUNC PyInit_simple_extension(void) {
    return PyModule_Create(&simplemodule);
}