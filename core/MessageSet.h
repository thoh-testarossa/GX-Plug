//
// Created by Thoh Testarossa on 2019-03-08.
//

#pragma once

#ifndef GRAPH_ALGO_MESSAGESET_H
#define GRAPH_ALGO_MESSAGESET_H

#include "../include/deps.h"

#define INVALID_MASSAGE INT32_MAX

class Message
{
public:
    Message(int src, int dst, double value);

    void print();

    int src;
    int dst;
    double value;
};

class MessageSet
{
public:
    MessageSet();
    void insertMsg(Message m);

    std::vector<Message> mSet;
};

#endif //GRAPH_ALGO_MESSAGESET_H
