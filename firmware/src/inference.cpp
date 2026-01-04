#include "inference.h"
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"


const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
const int kTensorArenaSize = 200 * 1024; 
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

extern "C" int inference_setup(void) {
    model = tflite::GetModel(model_universal);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printk("Model schema version mismatch\n");
        return -1;
    }

    static tflite::MicroMutableOpResolver<15> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddMul();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddMean();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
    model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printk("AllocateTensors() failed\n");
        return -1;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    return 0;
}

extern "C" int run_inference(float* spectrogram_data) {
    if (!interpreter) return -1;
    for (int i = 0; i < (66 * 129); i++) {
        float val = spectrogram_data[i];
        int8_t q_val = (int8_t)(val / input->params.scale + input->params.zero_point);
        input->data.int8[i] = q_val;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        printk("Invoke failed\n");
        return -1;
    }
    float probability = (output->data.int8[2] - output->params.zero_point) * output->params.scale;
    
    return (int)(probability * 100.0f);
}