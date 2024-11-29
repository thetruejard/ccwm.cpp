#include "ccwm.h"

#include <filesystem>
#include <iostream>



CCWM::CCWM(
    std::string model_path,
    bool verbose
) {
    this->model_path = model_path;
    this->verbose = verbose;
    if (!std::filesystem::exists(model_path)) {
        throw std::invalid_argument("Could not find model file: \"" + model_path + "\"");
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 100,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    this->context = ggml_init(params);


    std::cout << "Printing from C++: " << model_path << std::endl;
}

CCWM::~CCWM() {
    std::cout << "CCWM Destructor" << std::endl;
    ggml_free(this->context);
    std::cout << "Freed ggml" << std::endl;
}

