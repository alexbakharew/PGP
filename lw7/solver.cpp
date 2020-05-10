#include "solver.h"
#include <algorithm>
#include <iostream> 
#include <cstdio>
#include <vector>
#include <cstring>
namespace Jacobi
{

Solver::Solver(double _l_x, double _l_y, double _l_z,
                        int _block_size_x, int _block_size_y, int _block_size_z,
                        int _proc_count_x, int _proc_count_y, int _proc_count_z)
{
    block_size_x = _block_size_x;
    block_size_y = _block_size_y;
    block_size_z = _block_size_z;

    proc_count_x = _proc_count_x;
    proc_count_y = _proc_count_y;
    proc_count_z = _proc_count_z;

    total_cell_x = block_size_x * proc_count_x;
    total_cell_y = block_size_y * proc_count_y;
    total_cell_z = block_size_z * proc_count_z;

    h_x = _l_x / (double)(block_size_x * proc_count_x);
    h_y = _l_y / (double)(block_size_y * proc_count_y);
    h_z = _l_z / (double)(block_size_z * proc_count_z);
    //std::cout << h_x << " " << h_y << " " << h_z << std::endl;
    total_cell_count = total_cell_x * total_cell_y * total_cell_z;
    //mesh = std::shared_ptr<double>(new double[total_cell_count]);

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

bool Solver::InitInitialCondition(double _u_down, double _u_up, double _u_left, double _u_right, double _u_front, double _u_back, double _u_0)
{
    u_down =_u_down;
    u_up =_u_up;
    u_left = _u_left;
    u_right = _u_right;
    u_front = _u_front;
    u_back = _u_back;
    mesh = std::vector<double>(total_cell_count, _u_0);
}

bool Solver::Solve(const double eps)
{
    int i, j, k;
    
    double curr_eps;

    std::vector<double> new_mesh;
    double numerator, denominator;

    do
    {
        new_mesh = mesh;
        for(i = 0; i < total_cell_x; ++i)
        {
            for(j = 0; j < total_cell_y; ++j)
            {
                for(k = 0; k < total_cell_z; ++k)
                {
                    numerator = 0.0;

                    if(i == 0)
                    {
                        numerator += (mesh[_Get_i(i + 1, j, k)] +
                                u_left) / (h_x * h_x);
                    }
                    else if(i == total_cell_x - 1)
                    {
                        numerator += (u_right +
                                mesh[_Get_i(i - 1, j ,k)]) / (h_x * h_x);
                    }
                    else
                    {
                        numerator += (mesh[_Get_i(i + 1, j, k)] +
                                mesh[_Get_i(i - 1, j ,k)]) / (h_x * h_x);
                    }


                    if(j == 0)
                    {
                        numerator += (mesh[_Get_i(i, j + 1, k)] +
                                u_front) / (h_y * h_y);
                    }
                    else if(j == total_cell_y - 1)
                    {
                        numerator += (u_back +
                                mesh[_Get_i(i, j - 1, k)]) / (h_y * h_y);
                    }
                    else
                    {
                        numerator += (mesh[_Get_i(i, j + 1, k)] +
                                mesh[_Get_i(i, j - 1, k)]) / (h_y * h_y);
                    }


                    if(k == 0)
                    {
                        numerator += (mesh[_Get_i(i, j, k + 1)] +
                                u_down) / (h_z * h_z);
                    }
                    else if(k == total_cell_z - 1)
                    {
                        numerator += (u_up +
                                mesh[_Get_i(i, j, k - 1)]) / (h_z * h_z);
                    }
                    else
                    {
                        numerator += (mesh[_Get_i(i, j, k + 1)] +
                                mesh[_Get_i(i, j, k - 1)]) / (h_z * h_z);
                    }

                    denominator = 2 * ((1 / (h_x * h_x)) + (1 / (h_y * h_y)) + (1 / (h_z * h_z)));

                    double val = numerator / denominator;
                    new_mesh[_Get_i(i, j, k)] = val;
                }
            }
        }

        curr_eps = _CalcEps(new_mesh);

        mesh = new_mesh;

        std::cout << curr_eps << std::endl;
        //std::cout << "-----------------------" << std::endl;
    } while (curr_eps > eps);
    //} while (0);
    
    

}

bool Solver::SaveResults(const char* out_path) const
{
    FILE* output = fopen(out_path, "w+");
    double val;
    for(int i = 0; i < total_cell_count; ++i)
    {
        val = mesh[i];
        fprintf(output, "%e ", val);
    }
    fclose(output);
    return true;
}

void Solver::PrintMesh() const
{
    for(int i = 0; i < total_cell_count; ++i)
    {
        printf("%lf\n", mesh[i]);
    }
}

double Solver::_CalcEps(const std::vector<double>& new_mesh) const
{
    double max_sub = std::abs(mesh[0] - new_mesh[0]);
    double curr_sub;
    for(int i = 1; i < total_cell_count; ++i)
    {
       // std::cout << mesh.get()[i] << " " << new_mesh.get()[i] << std::endl;
        curr_sub = std::abs(mesh[i] - new_mesh[i]);
        if(curr_sub > max_sub)
            max_sub = curr_sub;
    }
    return max_sub;
}

int Solver::_Get_i(int x, int y, int z) const
{
    int i = (block_size_x * proc_count_x * block_size_y * proc_count_y) * z;
    i += (block_size_x * proc_count_x) * y;
    i += x;
    return i;
}

std::tuple<int, int, int> Solver::_Get_xyz(int i) const
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
