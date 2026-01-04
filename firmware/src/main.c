#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/drivers/gpio.h>
#include <stdlib.h>
#include "data.h"       
#include "dsp.h"        
#include "inference.h"  
#include "model.h"      


static const struct gpio_dt_spec led_status = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
static const struct gpio_dt_spec pin_bench = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios); 

static int16_t audio_buffer[16000] __attribute__((aligned(16)));
static float spectrogram_buffer[66 * 129] __attribute__((aligned(16)));



void run_injection_test(const char* name, const int16_t* source_data, bool is_silence, bool expect_match) {
    printk("\n--- TEST: %s ---\n", name);

    if (is_silence) {
        for(int i=0; i<16000; i++) audio_buffer[i] = (rand() % 10) - 5;
    } else {
        for(int i=0; i<NUM_SAMPLES; i++) audio_buffer[i] = source_data[i];
    }

    generate_spectrogram(audio_buffer, spectrogram_buffer);
    float max_vol = 0.0f;
    for(int i=0; i<(66*129); i++) {
        if(spectrogram_buffer[i] > max_vol) max_vol = spectrogram_buffer[i];
    }
    
    if (max_vol < 20.0f) {
        printk("Skipping AI: Signal too weak (Max: %d)\n", (int)max_vol);
        return; 
    }

    gpio_pin_set_dt(&pin_bench, 1);
    uint32_t t_start = k_cycle_get_32();
    int confidence = run_inference(spectrogram_buffer);
    
    uint32_t t_end = k_cycle_get_32();
    gpio_pin_set_dt(&pin_bench, 0);

    uint32_t duration_ms = k_cyc_to_ms_ceil32(t_end - t_start);
    printk("Inference: %d ms | Confidence: %d%%\n", duration_ms, confidence);

 
    if (confidence > 80) {
        gpio_pin_set_dt(&led_status, 1); // ON
        if (!expect_match) printk("False Positive\n");
        else printk("Target Detected\n");
    } else {
        gpio_pin_set_dt(&led_status, 0); // OFF
        if (expect_match) printk("Missed Target\n");
        else printk("Correctly Ignored\n");
    }
}

int main(void)
{
    if (!device_is_ready(led_status.port) || !device_is_ready(pin_bench.port)) return 0;
    gpio_pin_configure_dt(&led_status, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&pin_bench, GPIO_OUTPUT_ACTIVE);
    gpio_pin_set_dt(&led_status, 0);
    gpio_pin_set_dt(&pin_bench, 0);

    printk("*** nRF5340: KWS ***\n");
    if (inference_setup() != 0) {
        printk("AI Model Failed to Load\n");
        return 0;
    }
    printk("AI Initialized\n");

    while (1) {
        run_injection_test("Silence", NULL, true, false);
        k_msleep(2000);
        run_injection_test("Interference (Cat)", audio_cat2, false, false);
        k_msleep(2000);
        run_injection_test("Target (ON)", audio_on2, false, true);
        k_msleep(2000);
    }
    return 0;
}