#include <memory>
#include <iostream>
main()
{
    std::shared_ptr<double> a(new double[5]);
    for(int i = 0; i < 5; ++i)
    {
        a.get()[i] = i;
    }

    std::shared_ptr<double> b(new double[5]);

    std::copy(a.get(), a.get() + 5, b.get());

    for(int i = 0; i < 5; ++i)
    {
        a.get()[i] *= 10;
    }

    a.swap(b);

    for(int i = 0; i < 5; ++i)
    {
        std::cout << a.get()[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < 5; ++i)
    {
        std::cout << b.get()[i] << " ";
    }
    std::cout << std::endl;

    a.swap(b);

    for(int i = 0; i < 5; ++i)
    {
        std::cout << a.get()[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < 5; ++i)
    {
        std::cout << b.get()[i] << " ";
    }
    std::cout << std::endl;
}