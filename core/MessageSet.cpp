//
// Created by Thoh Testarossa on 2019-03-08.
//

#include "../core/MessageSet.h"

#include <iostream>

template <typename VertexValueType>
Message<VertexValueType>::Message(int src, int dst, const VertexValueType& value)
{
    this->src = src;
    this->dst = dst;
    this->value = value;
}

template <typename VertexValueType>
MessageSet<VertexValueType>::MessageSet()
{
    this->mSet = std::vector<Message<VertexValueType>>();
}

template <typename VertexValueType>
void MessageSet<VertexValueType>::insertMsg(const Message<VertexValueType>& m)
{
    this->mSet.push_back(m);
}
