//
// Created by kdy on 2021/3/13.
//

#include "ThreadPool.h"

ThreadPool::ThreadPool(int threadNum)
{
    if (threadNum > MAX_THREAD_NUM || threadNum < 0) threadNum = MAX_THREAD_NUM;

    this->threadNum = threadNum;
    this->idleThreadNum = threadNum;
    this->isRun = false;
}

ThreadPool::~ThreadPool()
{
    if (!this->isRun) stop();
}

void ThreadPool::work()
{
    while (this->isRun)
    {
        task t;
        {
            std::unique_lock<std::mutex> uniqueLock(this->lock);
            this->cond.wait(uniqueLock, [this] {
                return !this->taskQueue.empty() || !this->isRun;
            });
            if (this->taskQueue.empty()) break;
            t = taskQueue.front();
            taskQueue.pop();
        }
        this->idleThreadNum--;
        t();
        this->idleThreadNum++;
    }
    this->idleThreadNum++;
}

void ThreadPool::start()
{
    this->isRun = true;
    for (int i = 0; i < this->threadNum; i++)
        pool.emplace_back(std::thread(&ThreadPool::work, this));
}

void ThreadPool::stop()
{
    this->isRun = false;
    this->cond.notify_all();
    for (auto &thread : this->pool)
    {
        if (thread.joinable())
        {
            thread.join();
            this->idleThreadNum++;
        }
    }
    while (!this->taskQueue.empty()) this->taskQueue.pop();
}

void ThreadPool::commitTask(const task &t)
{
    if (!this->isRun) throw std::runtime_error("commit task when ThreadPool is stopped");
    {
        std::unique_lock<std::mutex> uniqueLock(this->lock);
        taskQueue.emplace(t);
    }
    this->cond.notify_one();
}

int ThreadPool::idleThreadCount()
{
    return this->idleThreadNum;
}