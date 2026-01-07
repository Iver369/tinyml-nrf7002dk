#include "dsp.h"
#include "data.h"          
#include "mel_constants.h" 

#include <zephyr/kernel.h>
#include <arm_math.h>      
#include <math.h>          

// aliogned(16) for SIMD instructions
// matching python training configuration
static float fft_input_buffer[FFT_SIZE] __attribute__((aligned(16)));
static float fft_output_buffer[FFT_SIZE] __attribute__((aligned(16)));
static float fft_magnitude_buffer[NUM_FFT_BINS] __attribute__((aligned(16)));

static float window_func[FFT_SIZE] __attribute__((aligned(16)));
static arm_rfft_fast_instance_f32 fft_instance;
static bool is_initialized = false;

void dsp_init(void) {
    if (!is_initialized) {
        arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);
        for (int i =0; i < FFT_SIZE; i++) {
            window_func[i] = 0.5f - 0.5f *arm_cos_f32(2.0f * PI * (float)i / (float)FFT_SIZE);
        }

        is_initialized = true;
    }
}

int generate_spectrogram(const int16_t* audio_in, float* features_out) {
    if (!is_initialized) dsp_init();

    int read_index = 0; 
    int write_index = 0; 

    for (int i = 0; i < SPECTROGRAM_ROWS; i++) {
        for (int j = 0; j < FFT_SIZE; j++) {
            if (read_index + j < NUM_SAMPLES) {
                // dividing by 32768 for neural network to work with ranges between -1 and 1
                float normalized = (float)audio_in[read_index + j] / 32768.0f;
                fft_input_buffer[j] = normalized * window_func[j]; // smoothing edges to avoid spectral leakage
            } else {
                fft_input_buffer[j] = 0.0f;
            }
        }
        /*
        the output is symmetrical because the input is real numbers (audio)
        so arm_rfft_fast_f32 deletes second half and then the dc and nyquist frequencies
        gets packed as they are purely real numbers
        */ 
        arm_rfft_fast_f32(&fft_instance, fft_input_buffer, fft_output_buffer, 0);

        float dc = fft_output_buffer[0];
        fft_magnitude_buffer[0] = (dc>= 0) ? dc : -dc; 
        float nyquist = fft_output_buffer[1];
        fft_magnitude_buffer[NUM_FFT_BINS - 1] = (nyquist >= 0) ? nyquist : -nyquist; 
        arm_cmplx_mag_f32(fft_output_buffer +2, fft_magnitude_buffer + 1, NUM_FFT_BINS - 2);

        for (int m = 0; m < NUM_MEL_BINS; m++) {
            float energy_sum = 0.0f;
            
            for (int k = 0; k < NUM_FFT_BINS; k++) {
                energy_sum += fft_magnitude_buffer[k] * mel_filterbank[k][m];
            }
            // adding 1e-6f to prevent log(0)
            features_out[write_index++] = logf(energy_sum + 1e-6f);
        }

        read_index += WINDOW_STEP;
    }

    return 0;
}