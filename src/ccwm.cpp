#include "ccwm.h"
#include <filesystem>


CCWM::CCWM(
    std::string model_path,
    bool verbose
) {
    this->model_path = model_path;
    this->verbose = verbose;
    if (!std::filesystem::exists(model_path)) {
        throw std::invalid_argument("Model file not found: " + model_path);
    }

    this->wrapper = std::make_unique<GGMLWrapper>(this->model_path, this->verbose);
}

std::string CCWM::get_model_path() {
    return this->model_path;
}
bool CCWM::get_verbose() {
    return this->verbose;
}
void CCWM::set_verbose(bool verbose) {
    this->verbose = verbose;
}

const nlohmann::json& CCWM::get_config() {
    if (this->wrapper)
        return this->wrapper->get_config();
    return {};
}

