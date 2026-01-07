//#include <stdio.h>
//#include <stdlib.h>

// 全局变量
int global_counter = 0;
const char* version = "1.0.0";

// 结构体定义
typedef struct {
    int x;
    int y;
    double value;
} Point;

typedef struct {
    char name[50];
    int age;
    Point location;
} Person;

// 函数声明
int add(int a, int b);
double calculate_distance(Point p1, Point p2);
void print_person(Person* p);

// 函数实现
int add(int a, int b) {
    return a + b;
}

double calculate_distance(Point p1, Point p2) {
    int dx = p1.x - p2.x;
    int dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

void print_person(Person* p) {
//    if (p) {
//        printf("Name: %s\n", p->name);
//        printf("Age: %d\n", p->age);
//        printf("Location: (%d, %d)\n", p->location.x, p->location.y);
//    }
}

// 内联函数
static inline int max(int a, int b) {
    return a > b ? a : b;
}

// 复杂结构体
typedef struct TreeNode {
    int data;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

// 递归函数
int tree_height(TreeNode* root) {
    if (!root) return 0;
    int left_height = tree_height(root->left);
    int right_height = tree_height(root->right);
    return 1 + (left_height > right_height ? left_height : right_height);
}