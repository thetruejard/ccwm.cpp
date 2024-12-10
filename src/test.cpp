#include "test.h"
#include <filesystem>


Test::Test(
    std::string model_path,
    bool verbose
) {
    this->model_path = model_path;
    this->verbose = verbose;
    if (!std::filesystem::exists(model_path)) {
        throw std::invalid_argument("Model file not found: " + model_path);
    }

    this->wrapper = std::make_unique<GGMLWrapper>(this->model_path, this->verbose);


    for (const auto& [key, val] : this->wrapper->get_tensors()) {
        this->log("Tensor: \"%s\"", key.c_str());
        std::string shape_str;
        for (int i = 0; i < ggml_n_dims(val); i++) {
            shape_str += std::to_string(val->ne[i]);
            if (i < ggml_n_dims(val)-1)
                shape_str += ",";
        }
        this->log("\tShape: [%s]", shape_str.c_str());
    }

}

std::string Test::get_model_path() {
    return this->model_path;
}
bool Test::get_verbose() {
    return this->verbose;
}
void Test::set_verbose(bool verbose) {
    this->verbose = verbose;
}

const nlohmann::json& Test::get_config() {
    if (this->wrapper)
        return this->wrapper->get_config();
    return {};
}

