#include "solver.h"
#include <algorithm>
#include <iostream> 
#include <cstdio>
namespace Jacobi
{

bool Solver::InitMesh(double _l_x, double _l_y, double _l_z, int _block_size_x, int _block_size_y, int _block_size_z)
{
    block_size_x = _block_size_x;
    block_size_y = _block_size_y;
    block_size_z = _block_size_z;
    h_x = _l_x / (double)block_size_x;
    h_y = _l_y / (double)block_size_y;
    h_z = _l_z / (double)block_size_z;
    total_cell_count = block_size_x * block_size_y * block_size_z;
    mesh = std::make_shared<double>(total_cell_count);
}

void get_pos(int block_size_x, int block_size_y, int& x_c, int& y_c, int& z_c, int i)
{
    z_c = i / (block_size_x * block_size_y);
    if(z_c != 0)
    {
        y_c = (i - z_c * block_size_x * block_size_y) / block_size_x;
        x_c = (i - z_c * block_size_x * block_size_y) % block_size_x;
    }
    else
    {
        y_c = i / block_size_x;
        x_c = i % block_size_x;
    }
}

bool Solver::InitInitialCondition(double u_down, double u_up, double u_left, double u_right, double u_front, double u_back, double u_0)
{
    int i;
    int x_c, y_c, z_c;

    for(i = 0; i < total_cell_count; ++i)
    {

        std::tie(x_c, y_c, z_c) = _GetPos(i);
        std::cout << x_c << " " << y_c << " " << z_c << std::endl;
        if(z_c == 0)
        {
            mesh.get()[i] = u_down;
        }
        else if(z_c == block_size_z - 1)
        {
            mesh.get()[i] = u_up;
        }
        else if(y_c == 0)
        {
            mesh.get()[i] = u_front;
        }
        else if(y_c == block_size_y - 1)
        {
            mesh.get()[i] = u_back;
        }
        else if(x_c == 0)
        {
            mesh.get()[i] = u_left;
        }
        else if(x_c == block_size_x - 1)
        {
            mesh.get()[i] = u_right;
        }
        else
        {
            mesh.get()[i] = u_0;
        }
    }
}

bool Solver::Solve(const double eps)
{
    int i, j, k;
    
    double curr_eps;
    do
    {
        for(i = 1; i < block_size_x - 1; ++i)
        {
            for(j = 1; j < block_size_y - 1; ++j)
            {
                for(k = 1; k < block_size_z - 1; ++k)
                {
                    
                }
            }
        }

        curr_eps = _CalcEps();

    } while (curr_eps > eps);
    
    

}

bool Solver::SaveResults(const char* out_path) const
{
    FILE* output = fopen(out_path, "w");
    bool status;
    for(int i = 0; i < total_cell_count; ++i)
    {
        if(fprintf(output, "%e ", mesh.get()[i]) == 1)
            status = true;
        else
        {
            status = false;
            printf("BAD AT: %f\n", mesh.get()[i]);
        }
    }
    fclose(output);
    return status;
}

void Solver::PrintMesh() const
{
    for(int i = 0; i < total_cell_count; ++i)
    {
        std::cout << mesh.get()[i] << std::endl;
    }
}


double Solver::_CalcEps() const
{
    return std::abs(*std::max_element(mesh.get(), mesh.get() + total_cell_count));
}

std::tuple<int, int, int> Solver::_GetPos(int i) const
{
    int x_c, y_c, z_c;
    z_c = i / (block_size_x * block_size_y);
    if(z_c != 0)
    {
        y_c = (i - z_c * block_size_x * block_size_y) / block_size_x;
        x_c = (i - z_c * block_size_x * block_size_y) % block_size_x;
    }
    else
    {
        y_c = i / block_size_x;
        x_c = i % block_size_x;
    }
    return std::make_tuple(x_c, y_c, z_c);
}

}