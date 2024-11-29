#include "ggmlwrapper.h"
#include "ggml-cpu.h"


GGMLWrapper::GGMLWrapper(std::string model_path, bool verbose) {
    this->model_path = model_path;
    this->verbose = verbose;

    //this->backend = ggml_backend_cpu_init();
    //this->log("Initialized GGML backend (CPU)");
    //
    //// TODO: Compute context size.
    //size_t ctx_size = 100;
    //
    //struct ggml_init_params params = {
    //    /*.mem_size   =*/ ctx_size,
    //    /*.mem_buffer =*/ NULL,
    //    // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    //    /*.no_alloc   =*/ true,
    //};
    //this->context = ggml_init(params);
    //this->log("Initialized GGML context (%s)", sizefmt(ctx_size).c_str());

    this->load_model();
}

GGMLWrapper::~GGMLWrapper() {
    if (this->context)
        ggml_free(this->context);
    if (this->gguf_ctx)
        gguf_free(this->gguf_ctx);
    if (this->backend)
        ggml_backend_free(this->backend);
    this->log("Freed GGML/GGUF contexts and backend");
}

const nlohmann::json& GGMLWrapper::get_config() {
    return this->config;
}

void GGMLWrapper::load_model() {

    gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &this->context,
    };
    this->gguf_ctx = gguf_init_from_file(this->model_path.c_str(), params);

    // kv
    {
        const int n_kv = gguf_get_n_kv(this->gguf_ctx);
        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(this->gguf_ctx, i);
            printf("%s: kv[%d]: key = %s\n", __func__, i, key);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(this->gguf_ctx);
        printf("%s: n_tensors: %d\n", __func__, n_tensors);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (this->gguf_ctx, i);
            const size_t offset = gguf_get_tensor_offset(this->gguf_ctx, i);
            printf("%s: tensor[%d]: name = %s, offset = %zu\n", __func__, i, name, offset);
        }
    }

    // data
    {
        const int n_tensors = gguf_get_n_tensors(this->gguf_ctx);
        for (int i = 0; i < n_tensors; ++i) {
            printf("%s: reading tensor %d data\n", __func__, i);
            const char * name = gguf_get_tensor_name(this->gguf_ctx, i);
            struct ggml_tensor * cur = ggml_get_tensor(this->context, name);
            printf("%s: tensor[%d]: n_dims = %d, name = %s, data = %p\n", __func__, i, ggml_n_dims(cur), cur->name, cur->data);
            // print first 10 elements
            const float * data = (const float *) cur->data;
            printf("%s data[:10] : ", name);
            for (int j = 0; j < std::min(int64_t(10), ggml_nelements(cur)); ++j) {
                printf("%f ", data[j]);
            }
            printf("\n\n");

            // check data
            if (false) {
                const float * data = (const float *) cur->data;
                for (int j = 0; j < ggml_nelements(cur); ++j) {
                    if (data[j] != 100 + i) {
                        fprintf(stderr, "%s: tensor[%d]: data[%d] = %f\n", __func__, i, j, data[j]);
                        gguf_free(this->gguf_ctx);
                    }
                }
            }
        }
    }

    printf("%s: ctx_data size: %zu\n", __func__, ggml_get_mem_size(this->context));


    this->parse_config();

}


void GGMLWrapper::parse_config() {

    this->config = {};

    const int n_kv = gguf_get_n_kv(this->gguf_ctx);
    for (int i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(this->gguf_ctx, i);
        gguf_type t = gguf_get_kv_type(this->gguf_ctx, i);
        switch (t) {
            case gguf_type::GGUF_TYPE_UINT8:   this->config[key] = gguf_get_val_u8  (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_UINT16:  this->config[key] = gguf_get_val_u16 (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_UINT32:  this->config[key] = gguf_get_val_u32 (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_UINT64:  this->config[key] = gguf_get_val_u64 (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_INT8:    this->config[key] = gguf_get_val_i8  (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_INT16:   this->config[key] = gguf_get_val_i16 (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_INT32:   this->config[key] = gguf_get_val_i32 (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_INT64:   this->config[key] = gguf_get_val_i64 (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_FLOAT32: this->config[key] = gguf_get_val_f32 (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_FLOAT64: this->config[key] = gguf_get_val_f64 (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_STRING:  this->config[key] = gguf_get_val_str (this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_BOOL:    this->config[key] = gguf_get_val_bool(this->gguf_ctx, i); break;
            case gguf_type::GGUF_TYPE_ARRAY: {
                this->config[key] = nlohmann::json::array();
                gguf_type arrt = gguf_get_arr_type(this->gguf_ctx, i);
                if (arrt != gguf_type::GGUF_TYPE_STRING) {
                    this->log("Warning: non-string array types are temporarily not supported. Skipping key: %s", key);
                    continue;
                }
                for (int j = 0; j < gguf_get_arr_n(this->gguf_ctx, i); j++) {
                    this->config[key].push_back(gguf_get_arr_str(this->gguf_ctx, i, j));
                }
            } break;
        }
    }

}


