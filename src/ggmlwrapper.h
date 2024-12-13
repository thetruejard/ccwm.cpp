#pragma once
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include "utils.h"
#include <string>
#include <iostream>
#include <unordered_map>


class GGMLWrapper {
public:

    GGMLWrapper(std::string model_path, bool verbose);
    ~GGMLWrapper();

    const nlohmann::json& get_config();

    const std::unordered_map<std::string, ggml_tensor*>& get_tensors();


private:

    std::string model_path;
    bool verbose;

    nlohmann::json config;
    std::unordered_map<std::string, ggml_tensor*> tensors;

    ggml_context* context = nullptr;
    gguf_context* gguf_ctx = nullptr;
    ggml_backend_t backend = nullptr;

    void load_model();      // Called by ctor, uses model_path
    void parse_config();    // Called by load_model, uses contexts
    void parse_tensors();   // Called by load_model, uses contexts

    template<typename... Args>
    void log(std::string msg, Args... args) {
        if (this->verbose) {
            std::cout << strfmt(msg, args...) << std::endl;
        }
    }

};
