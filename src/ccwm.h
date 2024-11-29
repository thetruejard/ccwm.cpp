#pragma once
#include "nlohmann/json.hpp"
#include "ggmlwrapper.h"
#include "utils.h"
#include <string>
#include <memory>


class CCWM {
public:

    CCWM(
        std::string model_path,
        bool verbose
    );
    
    std::string get_model_path();
    bool get_verbose();
    void set_verbose(bool verbose);

    const nlohmann::json& get_config();


private:

    std::string model_path;
    bool verbose;

    std::unique_ptr<GGMLWrapper> wrapper;

    template<typename... Args>
    void log(std::string msg, Args... args) {
        if (this->verbose) {
            std::cout << strfmt(msg, args...) << std::endl;
        }
    }

};
