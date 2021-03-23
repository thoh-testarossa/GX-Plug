//
// Created by kdy on 2021/3/10.
//
#ifndef GRAPH_ALGO_THREADPOOL_H
#define GRAPH_ALGO_THREADPOOL_H

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>

const int MAX_THREAD_NUM = 64;

class ThreadPool
{
public:
    using task = std::function<void()>;

    explicit ThreadPool(int threadNum);

    ~ThreadPool();

    void start();

    void stop();

    void commitTask(const task &t);

    int taskCount();

    int threadNum;

private:
    void work();

    std::vector<std::thread> pool;
    std::queue<task> taskQueue;
    std::mutex lock;
    std::condition_variable cond;
    std::atomic<bool> isRun{};
    std::atomic<int> idleTask{};
};

#endif //GRAPH_ALGO_THREADPOOL_H
