#include <iostream>
#include <memory>  //declarations of unique_ptr
// #include <string>

int main()
{
    std::shared_ptr<int> ptr1 {std::make_shared<int>(5)};
    int y = 4;
    std::shared_ptr<int> ptr2 {std::make_shared<int>(y)};
    
    return 0;
}
