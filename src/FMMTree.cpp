#include <H2_3D_Tree.hpp>
#include <kernel_Matern12.hpp>
#include <kernel_Matern32.hpp>
#include <kernel_Matern52.hpp>
#include <kernel_MaternInf.hpp>
#include <boost/python.hpp>
using namespace boost::python;


class BaseWrap : public H2_3D_Tree, public wrapper<H2_3D_Tree>
{
    public:
    BaseWrap(double L, int tree_level, int interpolation_order, double epsilon, int use_chebyshev):H2_3D_Tree(L,tree_level,interpolation_order, epsilon, use_chebyshev){}

    double EvaluateKernel(const vector3& targetpos, const vector3& sourcepos)
    {
         return this->get_override("EvaluateKernel")(boost::ref(targetpos), boost::ref(sourcepos));
    }
};


BOOST_PYTHON_MODULE(FMMTree)

{
    // https://docs.python.org/3/whatsnew/3.9.html
    Py_Initialize();

    class_<vector3>("vector3", init<double, double, double>())
    .def(init<>())
    .def_readwrite("x", &vector3::x)
    .def_readwrite("y", &vector3::y)
    .def_readwrite("z", &vector3::z);
    ;
    class_<kernel_Matern12, boost::noncopyable >("kernel_Matern12", init<double, int, int, double, int>())
    .def("buildFMMTree", &H2_3D_Tree::buildFMMTree)
    .def("EvaluateKernel", &H2_3D_Tree::EvaluateKernel)
    ;
    class_<kernel_Matern32, boost::noncopyable >("kernel_Matern32", init<double, int, int, double, int>())
    .def("buildFMMTree", &H2_3D_Tree::buildFMMTree)
    .def("EvaluateKernel", &H2_3D_Tree::EvaluateKernel)
    ;
    class_<kernel_Matern52, boost::noncopyable >("kernel_Matern52", init<double, int, int, double, int>())
    .def("buildFMMTree", &H2_3D_Tree::buildFMMTree)
    .def("EvaluateKernel", &H2_3D_Tree::EvaluateKernel)
    ;
    class_<kernel_MaternInf, boost::noncopyable >("kernel_MaternInf", init<double, int, int, double, int>())
    .def("buildFMMTree", &H2_3D_Tree::buildFMMTree)
    .def("EvaluateKernel", &H2_3D_Tree::EvaluateKernel)
    ;
   
    class_<BaseWrap, boost::noncopyable>("Atree", init<double, int, int, double, int>())
    .def("EvaluateKernel", pure_virtual(&H2_3D_Tree::EvaluateKernel))
    .def("buildFMMTree", &H2_3D_Tree::buildFMMTree)
    ;
}   
