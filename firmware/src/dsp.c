#include "dsp.h"
#include <arm_math.h> 
#include <zephyr/sys/printk.h>

// matching train_kws parameters
#define FRAME_LEN 256
#define FRAME_STEP 240
#define NUM_FRAMES 66
#define FFT_SIZE 256

int generate_spectrogram(const int16_t* waveform, float* spectrogram) {
static float frame_buffer[FRAME_LEN] __attribute__((aligned(16)));
static float fft_output[FFT_SIZE + 2] __attribute__((aligned(16)));
static arm_rfft_fast_instance_f32 fft_instance;
    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);
    for (int i = 0; i < NUM_FRAMES; i++) {
        for (int j = 0; j < FRAME_LEN; j++) {
            frame_buffer[j] = (float)waveform[i * FRAME_STEP + j] / 32768.0f;
        }

        arm_rfft_fast_f32(&fft_instance, frame_buffer, fft_output, 0);

        for (int j = 0; j < (FFT_SIZE / 2) + 1; j++) {
            float real = fft_output[j * 2];
            float imag = fft_output[j * 2 + 1];
            spectrogram[i * 129 + j] = sqrtf(real * real + imag * imag);
        }
    }
    return 0;
}