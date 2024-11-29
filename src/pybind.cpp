#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#define MACRO_STRINGIFY(x) #x
namespace py = pybind11;
#include "pybind11_json.hpp"

#include "ccwm.h"

class CCWMWrapper : public CCWM {
public:

    CCWMWrapper(
        const py::object& model_path,
        bool verbose
    ) : CCWM(
        py::str(model_path),
        verbose
    ) {}

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
        .def(
            py::init<const py::object&, bool>(),
            py::arg("model_path"),
            py::arg("verbose") = false
        )
        .def_property_readonly("model_path", &CCWMWrapper::get_model_path)
        .def_property("verbose", &CCWMWrapper::get_verbose, &CCWMWrapper::set_verbose)
        .def_property_readonly("config", &CCWMWrapper::get_config);



#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
