#include "data.h"
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/drivers/gpio.h>

static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios); 

int main(void)
{   if (!device_is_ready(led.port)) { \
    return 0; 
}
    k_msleep(1000); 
    printk("\n\n*** nRF7002 DK: SYSTEM ONLINE ***\n");
    printk("Hello World from nRF7002 DK!\n");
    printk("Value: %d\n", NUM_SAMPLES);
    printk("Audio Data Sample 8000: %d\n", audio_data[8000]); // verify to see if it matches
    gpio_pin_configure_dt(&led, GPIO_OUTPUT_ACTIVE);
    if (NUM_SAMPLES == 16000) { // lights if there are 16000 samples 
    printk("Turning on LED...\n");
    gpio_pin_set_dt(&led, 1); 
}
    while (1) {
        printk("Verification Value at Index 8000: %d\n", audio_data[8000]);
        
        k_msleep(2000);
    }
    return 0;
}