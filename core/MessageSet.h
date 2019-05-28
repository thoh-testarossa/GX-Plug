//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_MESSAGESET_H
#define GRAPH_ALGO_MESSAGESET_H

#include "../include/deps.h"

#define INVALID_MASSAGE INT32_MAX

template <typename VertexValueType>
class Message
{
public:
    Message(int src, int dst, const VertexValueType& value);

    int src;
    int dst;
    VertexValueType value;
};

template <typename VertexValueType>
class MessageSet
{
public:
    MessageSet();
    void insertMsg(const Message<VertexValueType>& m);

    std::vector<Message<VertexValueType>> mSet;
};

#endif //GRAPH_ALGO_MESSAGESET_H
