#include <vector>
#include <tuple>
namespace Jacobi
{

class Solver
{
public:
    Solver(){}
    virtual ~Solver(){}
    Solver(double _l_x, double _l_y, double _l_z,
                    int _block_size_x, int _block_size_y, int _block_size_z, 
                    int _proc_count_x, int _proc_count_y, int _proc_count_z);

    bool InitInitialCondition(double u_down, double u_up, double u_left, double u_right, double u_front, double u_back, double u_0);

    bool Solve(const double eps);
    bool SaveResults(const char* out_path) const;

    void PrintMesh() const;
private:
    double _CalcEps(const std::vector<double>& new_mesh) const;
    std::tuple<int, int, int> _Get_xyz(int i) const;
    int _Get_i(int x, int y, int z) const;
private: 
    std::vector<double> mesh;
    double h_x, h_y, h_z;
    int block_size_x, block_size_y, block_size_z;
    int proc_count_x, proc_count_y, proc_count_z;
    int total_cell_x, total_cell_y, total_cell_z;
    int total_cell_count;
    double u_down, u_up, u_left, u_right, u_front, u_back;

};

}
