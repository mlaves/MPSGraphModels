#include <iostream>
#include <ostream>

#include "MPSModelsCxx.h"

int main()
{
    MPSModelsCxx::test_mps();

    auto vgg16 = MPSModelsCxx::VGG16_C::init();
    auto prediction = vgg16.forward();

    return 0;
}
