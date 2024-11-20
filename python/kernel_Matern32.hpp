#include <H2_3D_Tree.hpp>
class kernel_Matern32: public H2_3D_Tree {
public:
    kernel_Matern32(double L, int tree_level, int interpolation_order, double epsilon, int use_chebyshev):H2_3D_Tree(L,tree_level,interpolation_order, epsilon, use_chebyshev){};
    void SetKernelProperty() {
        this->homogen = 0;
        this->symmetry = 1;
        this->kernelType = "Matern32";
    }
    double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos) {
        vector3 diff;
        diff.x = sourcepos.x - targetpos.x;
        diff.y = sourcepos.y - targetpos.y;
        diff.z = sourcepos.z - targetpos.z;
        double r = sqrt(
            3 * (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z)
        );
        return (1 + r) * exp(-r);
    };
};
