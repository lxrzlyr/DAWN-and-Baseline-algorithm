#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "cublas_v2.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include "access.h"
#include "function.h"
using namespace std;


int main()
{

    int n = 0;
    int select = 1;
    cout << "请输入矩阵的阶数" << endl;
    cin >> n;
    cout << "请选择是否图的类型：1、无权；2、有权"<<endl;
    cin >> select;
   
    long long** A = new long long* [n];
    long long** B = new long long* [n];
    long long** amt = new long long* [n];
    long long** len = new long long* [n];
    for (int i = 0; i < n; i++)
    {
        A[i] = new long long[n];
        B[i] = new long long[n];
        amt[i] = new long long[n];
        len[i] = new long long[n];
        for (int j = 0; j < n; j++)
        {
            A[i][j] = 0;
            B[i][j] = 0;
            amt[i][j] = 0;
            len[i][j] = 0;
        }
    }
    if (select == 1)
    {
        ifstream fin("matrix.txt");
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                fin >> A[i][j];
                if (A[i][j] == 1)
                {
                    B[i][j] =1;
                    amt[i][j] =1;
                    len[i][j] = 1;
                }
                else
                {
                    B[i][j] = 0;
                    amt[i][j] = -1;
                    len[i][j] = -1;
                }
            }
        cout << "程序已读入数据" << endl;
        unweighted(A, B, amt, len, n);
    }
    if (select == 2)
    {
        ifstream fin("matrix.txt");
        map<pair<int, int>, int>mp;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                fin >> A[i][j];

                if (A[i][j] == 1)
                {
                    B[i][j] = 1;
                    amt[i][j] = 1;
                    len[i][j] = 1;
                }
                else
                {
                    B[i][j] = 0;
                    amt[i][j] = -1;
                    len[i][j] = -1;
                    if (A[i][j] > 1)
                        mp[{i, j}] = A[i][j];
                }
            }
        cout << "程序已读入数据" << endl;
        weighted(A, B, amt, len, n,  mp);
    }
    ofstream fout2("length.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fout2 << len[i][j] << " ";
        fout2 << endl;
    }
    cout << "len数据打印完毕" << endl;
    ofstream fout("amount.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            fout << amt[i][j] << " ";
        fout << endl;
    }
    

    for (int i = 0; i < n; i++)
    {
        delete[]A[i];
        delete[]B[i];
        delete[]amt[i];
    }
    delete[]A;
    delete[]B;
    delete[]amt;



    return 0;
}

