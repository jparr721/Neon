//
// Created by jparr on 8/21/2021.
//

#include <iostream>
#include <solvers/LinearElastic.h>

auto LinearElastic::Solve() -> void {
    foobar.resize(10, 10);
    foobar.setRandom();
    std::cout << foobar << std::endl;
}
