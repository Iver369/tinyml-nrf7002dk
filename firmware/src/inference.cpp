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
//statically allocate memory, so that it doesn't crash
const int kTensorArenaSize = 165 * 1024; // 165 as AllocateTensors() kept failing for having too low buffer
static uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16))); 

static TfLiteTensor* model_input = nullptr;
static TfLiteTensor* model_output = nullptr;

extern "C" int inference_setup(void) {
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printk("Model schema version mismatch\n"); // checks if model trained in python is compatible with the tflm 
        return -1;
    }

    static tflite::MicroMutableOpResolver<15> micro_op_resolver; // using specific math functions to save memory rather than including all 
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
        printk("Arena too small for this model architecture\n");
        return -1;
    }

    model_input = interpreter->input(0);
    model_output = interpreter->output(0);

    if (model_input == nullptr || model_output == nullptr) {
        printk("Tensors are NULL\n");
        return -1; // stops program as it would crash
    }
    
    return 0;
}

extern "C" int run_inference(float* spectrogram_data) {
    if (model_input == nullptr || model_output == nullptr) return -1;
    int num_elements = model_input->bytes; 
    // quantization as the dsp speaks in floats 32 bit, and the ai model 8-bit integers
    // mapping numbers
    for (int i = 0; i <num_elements; i++) {
        float val = spectrogram_data[i];
        int8_t q_val = (int8_t)(val /model_input->params.scale + model_input->params.zero_point);
        model_input->data.int8[i] = q_val;
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        printk("Interpreter Invoke failed\n"); // check to see if the math ran correctly 
        return -1;
    }
    // dequantization, outputs an integer and converts back to float for probability
    float probability = (model_output->data.int8[2] - model_output->params.zero_point) * model_output->params.scale;
    // return probability so that main can handle it
    return (int)(probability * 100.0f);
}