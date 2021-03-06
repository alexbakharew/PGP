#include <stdio.h>
#include <memory>
#include "solver.h"
#include <iostream>
int main(int argc, char** argv)
{
    int proc_count_x, proc_count_y, proc_count_z;
    int block_size_x, block_size_y, block_size_z;
    char output_file[16];
    double eps;
    double l_x, l_y, l_z;
    double u_down, u_up, u_left, u_right, u_front, u_back;
    double u_0;

    scanf("%d %d %d", &proc_count_x, &proc_count_y, &proc_count_z);
    scanf("%d %d %d", &block_size_x, &block_size_y, &block_size_z);
    scanf("%s", output_file);
    scanf("%lf", &eps);
    scanf("%lf %lf %lf", &l_x, &l_y, &l_z);
    scanf("%lf %lf %lf %lf %lf %lf", &u_down, &u_up, &u_left, &u_right, &u_front, &u_back);
    scanf("%lf", &u_0);

    Jacobi::Solver s(l_x, l_y, l_z,
                block_size_x, block_size_y, block_size_z,
                proc_count_x, proc_count_y, proc_count_z);
                
    s.InitInitialCondition(u_down, u_up, u_left, u_right, u_front, u_back, u_0);
//    s.PrintMesh();
    s.Solve(eps);
//    std::cout << "Solved: " << std::endl;
//    s.PrintMesh();
     if(s.SaveResults(output_file))
         std::cout << "OK" << std::endl;
     else
         std::cout << "BAD" << std::endl;
    
}
