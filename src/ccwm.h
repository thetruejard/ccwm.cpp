
#include "ggml.h"

#include <string>


class CCWM {
public:

    CCWM(
        std::string model_path,
        bool verbose
    );
    ~CCWM();

private:

    std::string model_path;
    bool verbose;

    ggml_context* context;

};