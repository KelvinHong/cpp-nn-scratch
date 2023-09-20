#include <iostream>
// #include <memory>  //declarations of unique_ptr
// #include <string>

int* dummy()
{
    int* x = new int{5};
    return x;
}

int main()
{
    int* ptr { dummy() };
    std::cout << *ptr << '\n';
    delete ptr;
    return 0;
}
