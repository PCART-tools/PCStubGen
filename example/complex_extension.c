#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

// 自定义异常对象
static PyObject* ComplexError;

// 自定义Point结构体
typedef struct {
    PyObject_HEAD
    double x;
    double y;
} PointObject;

// Point构造函数
static PyObject* Point_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    PointObject* self;
    self = (PointObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->x = 0.0;
        self->y = 0.0;
    }
    return (PyObject*)self;
}

// Point初始化函数
static int Point_init(PointObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"x", "y", NULL};
    double x = 0.0, y = 0.0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|dd", kwlist, &x, &y)) {
        return -1;
    }
    
    self->x = x;
    self->y = y;
    return 0;
}

// Point成员访问
static PyMemberDef Point_members[] = {
    {"x", T_DOUBLE, offsetof(PointObject, x), 0, "X coordinate"},
    {"y", T_DOUBLE, offsetof(PointObject, y), 0, "Y coordinate"},
    {NULL}  // 哨兵值
};

// Point字符串表示
static PyObject* Point_repr(PointObject* self) {
    return PyUnicode_FromFormat("Point(%.2f, %.2f)", self->x, self->y);
}

// Point距离计算方法
static PyObject* Point_distance(PointObject* self, PyObject* args) {
    PointObject* other;
    
    if (!PyArg_ParseTuple(args, "O!", &PointType, &other)) {
        return NULL;
    }
    
    double dx = self->x - other->x;
    double dy = self->y - other->y;
    double distance = sqrt(dx * dx + dy * dy);
    
    return PyFloat_FromDouble(distance);
}

// Point方法定义
static PyMethodDef Point_methods[] = {
    {"distance", (PyCFunction)Point_distance, METH_VARARGS, "Calculate distance to another point"},
    {NULL}  // 哨兵值
};

// Point类型定义
static PyTypeObject PointType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "complex_extension.Point",
    .tp_doc = "Point objects with x, y coordinates",
    .tp_basicsize = sizeof(PointObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Point_new,
    .tp_init = (initproc)Point_init,
    .tp_members = Point_members,
    .tp_methods = Point_methods,
    .tp_repr = (reprfunc)Point_repr,
};

// 复杂的数学计算函数
static PyObject* complex_matrix_multiply(PyObject* self, PyObject* args) {
    PyObject* matrix_a;
    PyObject* matrix_b;
    
    // 解析参数
    if (!PyArg_ParseTuple(args, "OO", &matrix_a, &matrix_b)) {
        return NULL;
    }
    
    // 检查是否为列表
    if (!PyList_Check(matrix_a) || !PyList_Check(matrix_b)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be lists");
        return NULL;
    }
    
    // 获取矩阵维度
    Py_ssize_t rows_a = PyList_Size(matrix_a);
    Py_ssize_t cols_a = PyList_Size(PyList_GetItem(matrix_a, 0));
    Py_ssize_t rows_b = PyList_Size(matrix_b);
    Py_ssize_t cols_b = PyList_Size(PyList_GetItem(matrix_b, 0));
    
    // 检查矩阵维度是否匹配
    if (cols_a != rows_b) {
        PyErr_SetString(ComplexError, "Matrix dimensions do not match for multiplication");
        return NULL;
    }
    
    // 创建结果矩阵
    PyObject* result = PyList_New(rows_a);
    if (!result) return NULL;
    
    // 执行矩阵乘法
    for (Py_ssize_t i = 0; i < rows_a; i++) {
        PyObject* row = PyList_New(cols_b);
        if (!row) {
            Py_DECREF(result);
            return NULL;
        }
        PyList_SetItem(result, i, row);
        
        for (Py_ssize_t j = 0; j < cols_b; j++) {
            double sum = 0.0;
            for (Py_ssize_t k = 0; k < cols_a; k++) {
                PyObject* a_item = PyList_GetItem(PyList_GetItem(matrix_a, i), k);
                PyObject* b_item = PyList_GetItem(PyList_GetItem(matrix_b, k), j);
                
                double a_val = PyFloat_AsDouble(a_item);
                double b_val = PyFloat_AsDouble(b_item);
                
                if (a_val == -1.0 && PyErr_Occurred() || 
                    b_val == -1.0 && PyErr_Occurred()) {
                    Py_DECREF(result);
                    return NULL;
                }
                
                sum += a_val * b_val;
            }
            PyList_SetItem(row, j, PyFloat_FromDouble(sum));
        }
    }
    
    return result;
}

// 复杂的字符串处理函数
static PyObject* complex_text_analyzer(PyObject* self, PyObject* args) {
    const char* text;
    PyObject* keywords;
    
    if (!PyArg_ParseTuple(args, "sO!", &text, &PyList_Type, &keywords)) {
        return NULL;
    }
    
    // 创建结果字典
    PyObject* result = PyDict_New();
    if (!result) return NULL;
    
    // 统计字符数
    size_t text_len = strlen(text);
    PyDict_SetItemString(result, "char_count", PyLong_FromSize_t(text_len));
    
    // 统计单词数
    int word_count = 0;
    int in_word = 0;
    for (size_t i = 0; i < text_len; i++) {
        if (isalpha(text[i])) {
            if (!in_word) {
                word_count++;
                in_word = 1;
            }
        } else {
            in_word = 0;
        }
    }
    PyDict_SetItemString(result, "word_count", PyLong_FromLong(word_count));
    
    // 统计关键词出现次数
    PyObject* keyword_counts = PyDict_New();
    Py_ssize_t keywords_size = PyList_Size(keywords);
    
    for (Py_ssize_t i = 0; i < keywords_size; i++) {
        PyObject* keyword_obj = PyList_GetItem(keywords, i);
        const char* keyword = PyUnicode_AsUTF8(keyword_obj);
        if (!keyword) continue;
        
        int count = 0;
        const char* p = text;
        while ((p = strstr(p, keyword)) != NULL) {
            count++;
            p += strlen(keyword);
        }
        
        PyDict_SetItem(keyword_counts, keyword_obj, PyLong_FromLong(count));
    }
    PyDict_SetItemString(result, "keyword_counts", keyword_counts);
    
    return result;
}

// 复杂的数据结构处理函数
static PyObject* complex_data_aggregator(PyObject* self, PyObject* args) {
    PyObject* data_list;
    const char* group_by;
    const char* aggregate_func;
    
    if (!PyArg_ParseTuple(args, "Oss", &data_list, &group_by, &aggregate_func)) {
        return NULL;
    }
    
    if (!PyList_Check(data_list)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a list");
        return NULL;
    }
    
    // 创建分组结果字典
    PyObject* groups = PyDict_New();
    if (!groups) return NULL;
    
    Py_ssize_t data_size = PyList_Size(data_list);
    
    // 分组数据
    for (Py_ssize_t i = 0; i < data_size; i++) {
        PyObject* item = PyList_GetItem(data_list, i);
        if (!PyDict_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "List items must be dictionaries");
            Py_DECREF(groups);
            return NULL;
        }
        
        PyObject* key_obj = PyDict_GetItemString(item, group_by);
        if (!key_obj) {
            PyErr_SetString(ComplexError, "Group by key not found in item");
            Py_DECREF(groups);
            return NULL;
        }
        
        PyObject* group_list = PyDict_GetItem(groups, key_obj);
        if (!group_list) {
            group_list = PyList_New(0);
            PyDict_SetItem(groups, key_obj, group_list);
            Py_DECREF(group_list);
        }
        
        PyList_Append(group_list, item);
    }
    
    // 聚合数据
    PyObject* result = PyDict_New();
    if (!result) {
        Py_DECREF(groups);
        return NULL;
    }
    
    PyObject *key, *group_list;
    Py_ssize_t pos = 0;
    
    while (PyDict_Next(groups, &pos, &key, &group_list)) {
        Py_ssize_t group_size = PyList_Size(group_list);
        
        if (strcmp(aggregate_func, "count") == 0) {
            PyDict_SetItem(result, key, PyLong_FromSsize_t(group_size));
        }
        else if (strcmp(aggregate_func, "sum") == 0) {
            double total = 0.0;
            for (Py_ssize_t i = 0; i < group_size; i++) {
                PyObject* item = PyList_GetItem(group_list, i);
                PyObject* value_obj = PyDict_GetItemString(item, "value");
                if (value_obj) {
                    total += PyFloat_AsDouble(value_obj);
                }
            }
            PyDict_SetItem(result, key, PyFloat_FromDouble(total));
        }
        else if (strcmp(aggregate_func, "avg") == 0) {
            double total = 0.0;
            int count = 0;
            for (Py_ssize_t i = 0; i < group_size; i++) {
                PyObject* item = PyList_GetItem(group_list, i);
                PyObject* value_obj = PyDict_GetItemString(item, "value");
                if (value_obj) {
                    total += PyFloat_AsDouble(value_obj);
                    count++;
                }
            }
            if (count > 0) {
                PyDict_SetItem(result, key, PyFloat_FromDouble(total / count));
            } else {
                PyDict_SetItem(result, key, PyFloat_FromDouble(0.0));
            }
        }
    }
    
    Py_DECREF(groups);
    return result;
}

// 方法定义数组
static PyMethodDef ComplexMethods[] = {
    {"matrix_multiply", complex_matrix_multiply, METH_VARARGS, "Multiply two matrices"},
    {"text_analyzer", complex_text_analyzer, METH_VARARGS, "Analyze text and count keywords"},
    {"data_aggregator", complex_data_aggregator, METH_VARARGS, "Group and aggregate data"},
    {NULL, NULL, 0, NULL}  // 哨兵值
};

// 模块定义
static struct PyModuleDef complexmodule = {
    PyModuleDef_HEAD_INIT,
    "complex_extension",           // 模块名
    "A complex Python C extension example.", // 模块文档
    -1,                          // 模块状态大小
    ComplexMethods,               // 方法定义
    NULL,                        // 模块遍历函数
    NULL,                        // 清理函数
    NULL,                        // 模块支持函数
    NULL                         // 调试信息
};

// 模块初始化函数
PyMODINIT_FUNC PyInit_complex_extension(void) {
    PyObject* m;
    
    // 创建Point类型
    if (PyType_Ready(&PointType) < 0) {
        return NULL;
    }
    
    // 创建模块
    m = PyModule_Create(&complexmodule);
    if (m == NULL) {
        return NULL;
    }
    
    // 添加自定义异常
    ComplexError = PyErr_NewException("complex_extension.ComplexError", NULL, NULL);
    Py_XINCREF(ComplexError);
    if (PyModule_AddObject(m, "ComplexError", ComplexError) < 0) {
        Py_XDECREF(ComplexError);
        Py_CLEAR(ComplexError);
        Py_DECREF(m);
        return NULL;
    }
    
    // 添加Point类型到模块
    Py_INCREF(&PointType);
    if (PyModule_AddObject(m, "Point", (PyObject*)&PointType) < 0) {
        Py_XDECREF(&PointType);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}