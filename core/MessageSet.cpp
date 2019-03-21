//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "../core/MessageSet.h"

#include <iostream>

Message::Message(int src, int dst, double value)
{
    this->src = src;
    this->dst = dst;
    this->value = value;
}

void Message::print()
{
    std::cout << src << " -> " << dst << ": " << value << std::endl;
}

MessageSet::MessageSet()
{
    this->mSet = std::vector<Message>();
}

void MessageSet::insertMsg(Message m)
{
    this->mSet.push_back(m);
}
