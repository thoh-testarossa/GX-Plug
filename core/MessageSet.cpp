//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "../core/MessageSet.h"

#include <iostream>

template <typename T>
Message<T>::Message(int src, int dst, const T& value)
{
    this->src = src;
    this->dst = dst;
    this->value = value;
}

template <typename T>
MessageSet<T>::MessageSet()
{
    this->mSet = std::vector<Message<T>>();
}

template <typename T>
void MessageSet<T>::insertMsg(const Message<T>& m)
{
    this->mSet.push_back(m);
}
