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
//����˷�
// cuBLASʵ�־���˷����˷�������ڸ���B�����ֵ��
void matMult_cuBLAS1(long long**& A, long long**& B, int n, cublasHandle_t cuHandle)
{
    int rowSizeA = n;
    int rowSizeB = n;
    int colSizeA = n;
    int colSizeB = n;


    // 1.���ڴ���Ϊ��Ҫ����ľ��󿪱ٿռ�
    double* h_A = (double*)malloc(rowSizeA * colSizeA * sizeof(double));
    double* h_B = (double*)malloc(rowSizeB * colSizeB * sizeof(double));
    double* h_C = (double*)malloc(rowSizeA * colSizeB * sizeof(double));

    // 2.��ʼ���������h_A��h_B
    for (int i = 0; i < rowSizeA; i++)
        for (int j = 0; j < colSizeA; j++)
            h_A[i * colSizeA + j] = (double)A[i][j];
    for (int i = 0; i < rowSizeB; i++)
        for (int j = 0; j < colSizeB; j++)
            h_B[i * colSizeB + j] = (double)B[i][j];

    // 3.���Դ���Ϊ��Ҫ��������������󿪱ٿռ�
    double* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, rowSizeA * colSizeA * sizeof(double));
    cudaMalloc((void**)&d_B, rowSizeB * colSizeB * sizeof(double));
    cudaMalloc((void**)&d_C, rowSizeA * colSizeB * sizeof(double));

    // 4.��CPU���ݿ�����GPU��
    cublasSetVector(rowSizeA * colSizeA, sizeof(double), h_A, 1, d_A, 1);
    cublasSetVector(rowSizeB * colSizeB, sizeof(double), h_B, 1, d_B, 1);

    // 5.���ݽ�������˺����еĲ�����ִ�к˺������������
    double a = 1; double b = 0;
    cublasDgemm(cuHandle, CUBLAS_OP_T, CUBLAS_OP_T, rowSizeA, colSizeB, colSizeA, &a, d_A, colSizeA, d_B, colSizeB, &b, d_C, rowSizeA);

    // 6.��GPU��ȡ����������CPU��ȥ
    cublasGetVector(rowSizeA * colSizeB, sizeof(double), d_C, 1, h_C, 1);

    // 7.�������ֵ���������
    for (int i = 0; i < rowSizeA; i++)
        for (int j = 0; j < colSizeB; j++)
            B[i][j] = static_cast<long long>(h_C[j * rowSizeA + i]);

    // 8.�����ʹ�ù����ڴ�
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void unweighted(long long**& A, long long**& B, long long**& amt, long long**& len, int n)
{

    //����CPU��ʱ����,��ʼ��ʱ
    clock_t   start, finish;
    start = clock();

    //GPU��ʱ��ʼ
    cudaEvent_t Gstart, Gstop;
    cudaEventCreate(&Gstart);
    cudaEventCreate(&Gstop);
    cudaEventRecord(Gstart, 0);

    cublasHandle_t cuHandle;
    cublasStatus_t status = cublasCreate(&cuHandle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            cout << "CUBLAS ����ʵ��������" << endl;
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

    //GPU��ʱ����
    cudaEventRecord(Gstop, 0);
    cudaEventSynchronize(Gstop);
    float elapsedTime;//�¼�ʱ��
    cudaEventElapsedTime(&elapsedTime, Gstart, Gstop);


    //����CPU��ʱ
    finish = clock();


    //��ӡ��������ʱ��
    cout << "CPU��ʱ������������ʱ�䣩Ϊ��" << double(finish - start) / 1000 << "s" << endl;
    cout << "GPU��ʱ������������ʱ�䣩Ϊ��" << elapsedTime / 1000 << "s" << endl;
}

void weighted(long long**& A, long long**& B, long long**& amt, long long**& len, int n, map<pair<int, int>, int>mp)
{
    //����CPU��ʱ����,��ʼ��ʱ
    clock_t   start, finish;
    start = clock();

    //GPU��ʱ��ʼ
    cudaEvent_t Gstart, Gstop;
    cudaEventCreate(&Gstart);
    cudaEventCreate(&Gstop);
    cudaEventRecord(Gstart, 0);

    cublasHandle_t cuHandle;
    cublasStatus_t status = cublasCreate(&cuHandle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            cout << "CUBLAS ����ʵ��������" << endl;
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


    //GPU��ʱ����
    cudaEventRecord(Gstop, 0);
    cudaEventSynchronize(Gstop);
    float elapsedTime;//�¼�ʱ��
    cudaEventElapsedTime(&elapsedTime, Gstart, Gstop);


    //����CPU��ʱ
    finish = clock();


    //��ӡ��������ʱ��
    cout << "CPU��ʱ������������ʱ�䣩Ϊ��" << double(finish - start) / 1000 << "s" << endl;
    cout << "GPU��ʱ������������ʱ�䣩Ϊ��" << elapsedTime / 1000 << "s" << endl;
}

void optfun(long long**& amt, long long**& len, int n, int x,int y,int dim)//ֻ�ܼ�����룬���ܻ�ȡ·������
{
    map<int, int>mp;
    int min = -1;
    int unc = 0;
    for (int i = 0; i < n; i++)
    {
        //��һ�������x,yλ�ö��ѳ���
        //�ڶ��������x��yδ����
        //�����������x��y��δ����
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
        cout << "���ִ���������뼰����" << endl;
    if (unc == 0)//�����ڵ㲻����
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

