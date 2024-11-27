#include <pybind11/pybind11.h>
#define MACRO_STRINGIFY(x) #x
namespace py = pybind11;

#include "ccwm.h"

class CCWMWrapper : public CCWM {
public:

    CCWMWrapper(
        const py::object& model_path
    ) : CCWM(py::str(model_path)) {}

};

PYBIND11_MODULE(ccwm_cpp, m) {
    m.doc() = R"pbdoc(
        CCWM
        -----------------------

        .. currentmodule:: ccwm_cpp

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    py::class_<CCWMWrapper>(m, "CCWM")
        .def(py::init<const py::object&>());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
