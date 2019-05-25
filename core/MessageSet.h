//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_MESSAGESET_H
#define GRAPH_ALGO_MESSAGESET_H

#include "../include/deps.h"

#define INVALID_MASSAGE INT32_MAX

template <typename T>
class Message
{
public:
    Message(int src, int dst, const T& value);

    int src;
    int dst;
    T value;
};

template <typename T>
class MessageSet
{
public:
    MessageSet();
    void insertMsg(const Message<T>& m);

    std::vector<Message<T>> mSet;
};

#endif //GRAPH_ALGO_MESSAGESET_H
