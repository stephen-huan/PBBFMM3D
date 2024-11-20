#include <compute.hpp>
#include "bbfmm.h"
#include <kernel_Matern12.hpp>
#include <kernel_Matern32.hpp>
#include <kernel_Matern52.hpp>
#include <kernel_MaternInf.hpp>
#include <H2_3D_Tree.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;


void convert_to_numpy(const vector<double> & input, object&  obj)
{
    PyObject* pobj = obj.ptr();
    Py_buffer pybuf;
    PyObject_GetBuffer(pobj, &pybuf, PyBUF_SIMPLE);
    void *buf = pybuf.buf;
    double *p = (double*)buf;
    Py_XDECREF(pobj);

    for (size_t i  = 0; i < input.size(); i++)
    {
        p[i] = input[i];
    }
}

void convert_to_vecOfdouble(const object& obj, vector<double>& output)
{
    // output is in colwise major
    PyObject* pobj = obj.ptr();
    Py_buffer pybuf;
    PyObject_GetBuffer(pobj, &pybuf, PyBUF_SIMPLE);
    void *buf = pybuf.buf;
    double *p = (double*)buf;
    Py_XDECREF(pobj);
    output.resize(len(obj));
    for (int i = 0; i < len(obj); i++)
      output[i] = p[i];
}

void convert_to_vecOfvec3(const object& obj, vector<vector3>& output)
{
    PyObject* pobj = obj.ptr();
    Py_buffer pybuf;
    PyObject_GetBuffer(pobj, &pybuf, PyBUF_SIMPLE);
    void *buf = pybuf.buf;
    double *p = (double*)buf;
    Py_XDECREF(pobj);
    output.resize(len(obj));

    for (int i  = 0; i < len(obj); i++)
    {
        output[i].x = p[i*3+0];
        output[i].y = p[i*3+1];
        output[i].z = p[i*3+2];
    }
}


BOOST_PYTHON_MODULE(FMMCompute)
{
   // https://docs.python.org/3/whatsnew/3.9.html
   Py_Initialize();
   class_<std::vector<vector3> >("vecOfvec3")
    .def(vector_indexing_suite<std::vector<vector3> >())
    ;
   class_<std::vector<double> >("vecOfdouble")
    .def(vector_indexing_suite<std::vector<double> >())
    ;
   class_<H2_3D_Compute<kernel_Matern12>, boost::noncopyable>("ComputeMatern12", init<kernel_Matern12& , std::vector<vector3>& , std::vector<vector3>&,  std::vector<double>& ,int , std::vector<double>& >())
   .def(init<kernel_Matern12& , std::vector<vector3>& , std::vector<vector3>&, std::vector<double>& ,int , std::vector<double>& >());
   class_<H2_3D_Compute<kernel_Matern32>, boost::noncopyable>("ComputeMatern32", init<kernel_Matern32& , std::vector<vector3>& , std::vector<vector3>&,  std::vector<double>& ,int , std::vector<double>& >())
   .def(init<kernel_Matern32& , std::vector<vector3>& , std::vector<vector3>&, std::vector<double>& ,int , std::vector<double>& >());
   class_<H2_3D_Compute<kernel_Matern52>, boost::noncopyable>("ComputeMatern52", init<kernel_Matern52& , std::vector<vector3>& , std::vector<vector3>&,  std::vector<double>& ,int , std::vector<double>& >())
   .def(init<kernel_Matern52& , std::vector<vector3>& , std::vector<vector3>&, std::vector<double>& ,int , std::vector<double>& >());
   class_<H2_3D_Compute<kernel_MaternInf>, boost::noncopyable>("ComputeMaternInf", init<kernel_MaternInf& , std::vector<vector3>& , std::vector<vector3>&,  std::vector<double>& ,int , std::vector<double>& >())
   .def(init<kernel_MaternInf& , std::vector<vector3>& , std::vector<vector3>&, std::vector<double>& ,int , std::vector<double>& >());
   def("convert_to_numpy", convert_to_numpy);
   def("convert_to_vecOfdouble", convert_to_vecOfdouble);
   def("convert_to_vecOfvec3", convert_to_vecOfvec3);
}   

