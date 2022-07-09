#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "access.h"
#include "cublas_v2.h"
#include <time.h>
#include <omp.h>
using namespace std;



void matMult_cuBLAS1(long long**& A, long long**& B, int n, cublasHandle_t cuHandle);
void unweighted(long long**& A, long long**& B, long long**& amt, long long**& len, int n);
void weighted(long long**& A, long long**& B, long long**& atm, long long**& len, int n, map<pair<int, int>, int>mp);
//矩阵乘法
// cuBLAS实现矩阵乘法（乘法结果用于更新B矩阵的值）
void matMult_cuBLAS1(long long**& A, long long**& B, int n, cublasHandle_t cuHandle)
{
    int rowSizeA = n;
    int rowSizeB = n;
    int colSizeA = n;
    int colSizeB = n;


    // 1.在内存中为将要计算的矩阵开辟空间
    double* h_A = (double*)malloc(rowSizeA * colSizeA * sizeof(double));
    double* h_B = (double*)malloc(rowSizeB * colSizeB * sizeof(double));
    double* h_C = (double*)malloc(rowSizeA * colSizeB * sizeof(double));

    // 2.初始化计算矩阵h_A和h_B
    for (int i = 0; i < rowSizeA; i++)
        for (int j = 0; j < colSizeA; j++)
            h_A[i * colSizeA + j] = (double)A[i][j];
    for (int i = 0; i < rowSizeB; i++)
        for (int j = 0; j < colSizeB; j++)
            h_B[i * colSizeB + j] = (double)B[i][j];

    // 3.在显存中为将要计算矩阵与结果矩阵开辟空间
    double* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, rowSizeA * colSizeA * sizeof(double));
    cudaMalloc((void**)&d_B, rowSizeB * colSizeB * sizeof(double));
    cudaMalloc((void**)&d_C, rowSizeA * colSizeB * sizeof(double));

    // 4.将CPU数据拷贝到GPU上
    cublasSetVector(rowSizeA * colSizeA, sizeof(double), h_A, 1, d_A, 1);
    cublasSetVector(rowSizeB * colSizeB, sizeof(double), h_B, 1, d_B, 1);

    // 5.传递进矩阵相乘函数中的参数并执行核函数，矩阵相乘
    double a = 1; double b = 0;
    cublasDgemm(cuHandle, CUBLAS_OP_T, CUBLAS_OP_T, rowSizeA, colSizeB, colSizeA, &a, d_A, colSizeA, d_B, colSizeB, &b, d_C, rowSizeA);

    // 6.从GPU中取出运算结果至CPU中去
    cublasGetVector(rowSizeA * colSizeB, sizeof(double), d_C, 1, h_C, 1);

    // 7.将结果赋值给结果矩阵
    for (int i = 0; i < rowSizeA; i++)
        for (int j = 0; j < colSizeB; j++)
            B[i][j] = static_cast<long long>(h_C[j * rowSizeA + i]);

    // 8.清理掉使用过的内存
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void unweighted(long long**& A, long long**& B, long long**& amt, long long**& len, int n)
{

    //设置CPU计时函数,开始计时
    clock_t   start, finish;
    start = clock();

    //GPU计时开始
    cudaEvent_t Gstart, Gstop;
    cudaEventCreate(&Gstart);
    cudaEventCreate(&Gstop);
    cudaEventRecord(Gstart, 0);

    cublasHandle_t cuHandle;
    cublasStatus_t status = cublasCreate(&cuHandle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            cout << "CUBLAS 对象实例化出错" << endl;
        }
        getchar();
    }
    int k = 0;
    int tmp = 0;
    int dim = 1;
    int km = n * (n - 1);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            if (len[i][j] == 1 )
                k++;
        }
    while (1)
    {
        dim++;
        matMult_cuBLAS1(A, B, n, cuHandle);
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (B[i][j] != 0 && len[i][j] == -1 && i != j)
                {
                    amt[i][j] = B[i][j];
                    len[i][j] = dim;
                    k++;
                    if (k > km - 1)
                        break;
                }
            }
            if (k > km - 1)
                break;
        }
        if (tmp == k)
            break;
        tmp = k;
        cout << dim << endl;
        cout << k << endl;
    }

    //GPU计时结束
    cudaEventRecord(Gstop, 0);
    cudaEventSynchronize(Gstop);
    float elapsedTime;//事件时间
    cudaEventElapsedTime(&elapsedTime, Gstart, Gstop);


    //结束CPU计时
    finish = clock();


    //打印程序运行时间
    cout << "CPU计时（程序运行总时间）为：" << double(finish - start) / 1000 << "s" << endl;
    cout << "GPU计时（程序运行总时间）为：" << elapsedTime / 1000 << "s" << endl;
}

void weighted(long long**& A, long long**& B, long long**& amt, long long**& len, int n, map<pair<int, int>, int>mp)
{
    //设置CPU计时函数,开始计时
    clock_t   start, finish;
    start = clock();

    //GPU计时开始
    cudaEvent_t Gstart, Gstop;
    cudaEventCreate(&Gstart);
    cudaEventCreate(&Gstop);
    cudaEventRecord(Gstart, 0);

    cublasHandle_t cuHandle;
    cublasStatus_t status = cublasCreate(&cuHandle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            cout << "CUBLAS 对象实例化出错" << endl;
        }
        getchar();
    }
    int k = 0;
    int tmp = 0;
    int dim = 1;
    int km = n * (n - 1);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            A[i][j] = B[i][j];
            if (len[i][j] == 1)
                k++;
        }

    while (1)
    {
        dim++;
        matMult_cuBLAS1(A, B, n, cuHandle);
        //for (int i = 0; i < n; i++)
        //{
        //    for (int j = 0; j < n; j++)
        //    {
        //        B[i][j] = max(B[i][j], B[j][i]);
        //        
        //        cout << B[i][j] << " ";
        //    }
        //        
        //    cout << endl;
        //}
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (mp[{i, j}] == dim)
                    B[i][j]++;
                if (B[i][j] != 0 && len[i][j] == -1 && i != j)
                { 
                    amt[i][j] = B[i][j];
                    len[i][j] = dim;
                    k++;
                    if (k > km - 1)
                        break;
                }
            }
            if (k > km - 1)
                break;
        }
        if (k > km - 1)
            break;
        if (tmp == k)
            break;
        tmp = k;
        cout << dim << endl;
        cout << k << endl;
        //cout << dim << endl;
        //cout << k << endl;
        //cout << "matrix C" << endl;
        //for (int i = 0; i < n; i++)
        //{
        //    for (int j = 0; j < n; j++)
        //        cout << C[i][j] << " ";
        //    cout << endl;
        //}
        //cout << "matrix D" << endl;
        //for (int i = 0; i < n; i++)
        //{
        //    for (int j = 0; j < n; j++)
        //        cout << D[i][j] << " ";
        //    cout << endl;
        //}
    }


    //GPU计时结束
    cudaEventRecord(Gstop, 0);
    cudaEventSynchronize(Gstop);
    float elapsedTime;//事件时间
    cudaEventElapsedTime(&elapsedTime, Gstart, Gstop);


    //结束CPU计时
    finish = clock();


    //打印程序运行时间
    cout << "CPU计时（程序运行总时间）为：" << double(finish - start) / 1000 << "s" << endl;
    cout << "GPU计时（程序运行总时间）为：" << elapsedTime / 1000 << "s" << endl;
}

void optfun(long long**& amt, long long**& len, int n, int x,int y,int dim)//只能计算距离，不能获取路径条数
{
    map<int, int>mp;
    int min = -1;
    int unc = 0;
    for (int i = 0; i < n; i++)
    {
        //第一种情况：x,y位置都已出现
        //第二种情况：x或y未出现
        //第三种情况：x和y都未出现
        if (len[x][i] != -1 && len[i][y] != -1)
        {
            int tmp = len[x][i] + len[i][y];
            if (min == -1||min > tmp)
                min = tmp; 
            mp[tmp]++;
            unc++;
        } 
    }
    if (min <= dim)
        cout << "出现错误，请检查代码及数据" << endl;
    if (unc == 0)//两个节点不相连
    {
        len[x][y] = -2;
        amt[x][y] = -2;
    }
    else
    {
        len[x][y] = min;
        amt[x][y] = mp[min];
    }
    mp.clear();
}

