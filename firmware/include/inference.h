#ifndef INFERENCE_H
#define INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif
int inference_setup(void);
int run_inference(float* spectrogram_data);
#ifdef __cplusplus
}
#endif

#endif 