#include <stdio.h>
#include <time.h>
#include <vector>

#define ARRAY_SIZE 100000000

// 未展开的循环
void normal_loop(int *arr) {
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] *= 2;
    }
}

// 展开的循环
void unrolled_loop(int *arr) {
    int i;
    for (i = 0; i < ARRAY_SIZE - 3; i += 4) {
        arr[i] *= 2;
        arr[i + 1] *= 2;
        arr[i + 2] *= 2;
        arr[i + 3] *= 2;
    }
    // 处理剩余的元素
    for (; i < ARRAY_SIZE; i++) {
        arr[i] *= 2;
    }
}

// 计时函数
double measure_time(void (*func)(int *), int *arr) {
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    func(arr);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    return cpu_time_used;
}

int main() {
    std::vector<int> arr1(ARRAY_SIZE);
    std::vector<int> arr2(ARRAY_SIZE);

    // 初始化数组
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr1[i] = i;
        arr2[i] = i;
    }

    // 测量未展开循环的时间
    double normal_time = measure_time(normal_loop, arr1.data());
    // 测量展开循环的时间
    double unrolled_time = measure_time(unrolled_loop, arr2.data());

    printf("Normal loop time: %f seconds\n", normal_time);
    printf("Unrolled loop time: %f seconds\n", unrolled_time);

    return 0;
}
