#include "access.h"
#include <iostream>             //Dijkstra二叉堆优化算法
#include <queue>
#include <string.h>
#include <time.h>
#include <omp.h>

using namespace std;
void Dijkstra_heap(int**& len, int n);
void Dijkstra( int**& len ,int n );
void Dijkstra_queue(int**& len, int n);

struct Node
{
    int s, w;

    bool operator < (const Node& t) const
    {
        return t.w < w;
    }
};

struct Node_queue {
    int u, step;
    Node_queue() {};
    Node_queue(int a, int sp)
    {
        u = a, step = sp;
    }
    bool operator<(const Node_queue& a)const {//重载 <
        return step > a.step;
    }
};


void Dijkstra_heap( int **& len,int n)
{
    int soc = 0;
    int INF = 214748364;
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        int* dist = new int[n];
        soc = i;
        priority_queue<Node> p;
        for (int i = 0; i < n; i++)
            dist[i] = INF;
        dist[soc] = 0;
        p.push({ soc,0 });
        while (!p.empty())
        {
            Node q = p.top();
            p.pop();
            if (dist[q.s] < q.w)
            {
                continue;
            }

            for (int j = 0; j < n; j++)
            {
                if (j == i)
                    continue;
                if (dist[j] > len[q.s][j] + q.w)     //dis数组数值更新（核心思想）
                {
                    dist[j] = len[q.s][j] + q.w;
                    p.push({ j,dist[j] });    //将周围点与它们更新后的dis数值入队
                    len[i][j] = dist[j];
                }
            }

        }

        priority_queue<Node> null_queue;
        p.swap(null_queue);
    }
    
}

void Dijkstra_queue(int**& len, int n)
{
    int soc = 2;
    int INF = 214748364;
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        soc = i;
        priority_queue< Node_queue >Q;//优先队列优化
        Q.push(Node_queue(soc, 0));
        int* dist = new int[n];
        int* flag = new int[n];
        for (int j = 0; j < n; j++)
        {
            dist[j] = INF;//初始化所有距离为无穷大
            flag[j] = 0;
        }

        dist[soc] = 0;
        while (!Q.empty())
        {
            Node_queue  it = Q.top();//优先队列列头元素为最小值
            Q.pop();
            int t = it.u;
            if (flag[t])//说明已经找到了最短距离，该节点是队列里面的重复元素
                continue;
            flag[t] = 1;
            for (int j = 0; j < n; j++)
            {
                if (!flag[j] && len[t][j] < INF)//判断与当前点有关系的点，并且自己不能到自己
                    if (dist[j] > dist[t] + len[t][j])
                    {
                        //求距离当前点的每个点的最短距离，进行松弛操作
                        dist[j] = dist[t] + len[t][j];
                        Q.push(Node_queue(j, dist[j]));//把更新后的最短距离压入队列中，注意：里面有重复元素
                        len[i][j] = dist[j];
                    }
            }
        }

        priority_queue<Node_queue> null_queue;
        Q.swap(null_queue);
    }

}

void Dijkstra( int**& len, int n)
{
    int soc = 0;
    for (int i = 0; i < n; i++)
    {
        soc = i;
        int* dis = new int[n];
        int* vis = new int[n];
        for (int i = 0; i < n; i++)
        {
            dis[i] = len[soc][i];
            vis[i] = 0;
        }

        dis[soc] = 0;
        vis[soc] = 1;
        int next = 0;

        for (int k = 0; k < n; k++)
        {
            int max = 214748364;

            for (int j = 0; j < n; j++)
            {
                if (vis[j] == 0 && max > dis[j])
                {
                    max = dis[j];
                    next = j;

                }
            }

            vis[next] = 1;

            for (int j = 0; j < n; j++)
            {
                if (vis[j] == 0 && dis[j] > len[next][j] + dis[next])
                {
                    dis[j] = len[next][j] + dis[next];
                }
            }
        }
        for (int j = 0; j < n; j++)
            len[soc][j]=dis[j];
    }
}



int main()
{
    int n = 0;
    cout << "请输入节点数量" << endl;
    cin >> n;


    int INF = 214748364;

    int **len= new int* [n];
    for (int i = 0; i < n; i++)
        len[i] = new int [n];
    ifstream fin("matrix.txt");
    for (int i = 0; i < n; i++)       //将初始点到其他点的最短路初始化为∞
    {
        for (int j = 0; j < n; j++)
        {
            int tmp = 0;
            fin >> tmp;
            if (tmp == 0)
                len[i][j] = INF;
            else
                len[i][j] = tmp;
        }
    }
    cout << "数据初始化完成，开始计算路径" << endl;
   //设置CPU计时函数,开始计时
    clock_t   start, finish;
    start = clock();
    Dijkstra(len, n);
    Dijkstra_queue( len,n);
    Dijkstra_heap(len, n);
    //结束CPU计时
    finish = clock();
    //打印程序运行时间
    cout << "dijkstra算法的运行时间为：" << double(finish - start) / 1000 << "s" << endl;

    ofstream fout("lenght.txt");
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                len[i][j] = 0;
                fout << len[i][j] << " ";
            }
            else
            {
                if (len[i][j] == INF)
                {
                    len[i][j] = 0;
                    fout << len[i][j] << " ";
                }
                else
                    fout << len[i][j] << " ";
            }
 
        }
        fout << endl;
    }
    return 0;

}
