#include <memory>
#include <tuple>
namespace Jacobi
{

class Solver
{
public:
    Solver(){}
    virtual ~Solver(){}
    bool InitMesh(double _l_x, double _l_y, double _l_z, int _block_size_x, int _block_size_y, int _block_size_z);

    bool InitInitialCondition(double u_down, double u_up, double u_left, double u_right, double u_front, double u_back, double u_0);

    bool Solve(const double eps);
    bool SaveResults(const char* out_path) const;

    void PrintMesh() const;
private:
    double _CalcEps() const;
    std::tuple<int, int, int> _GetPos(int i) const;
private: 
    std::shared_ptr<double> mesh;
    double h_x, h_y, h_z;
    int block_size_x, block_size_y, block_size_z;
    int total_cell_count;

};

}