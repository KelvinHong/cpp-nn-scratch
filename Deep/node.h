#ifndef NODE_H
#define NODE_H
#include "base.h"
#include <vector>

namespace Deep
{

template <typename Datatype>
class Node
{
    private:
        Datatype data;
        std::vector<std::pair<Deep::Layer*, int>> nextFunctions;
    public:
        void backward();
};

}

#endif