#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/drivers/gpio.h>
#include <stdlib.h>

#include "data.h"       
#include "dsp.h"        
#include "inference.h"  
#include "model.h"     
#include <hal/nrf_cache.h> 

#define CONFIDENCE_THRESHOLD 75 // setting to determine how confident it has to be to take action
#define SIGNAL_NOISE_FLOOR  2.0f 

#ifdef ENABLE_WIFI_TELEMETRY // macro for later when I add WiFi
    #include <zephyr/net/net_if.h>
    #include <zephyr/net/wifi_mgmt.h>
    #include <zephyr/net/net_event.h>
    #include <zephyr/net/net_mgmt.h>
    #include <zephyr/net/socket.h>
    #include "wifi_credentials.h"
#endif

// device tree with zephyr to automatically find the gpios
static const struct gpio_dt_spec led_target = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios); // indication of detected correct sound
static const struct gpio_dt_spec pin_debug  = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios); // indication of inference

// static and global variables to avoid putting on stack, and using aligned(16) for the ARM Cortex-M33 processor
int16_t audio_buffer[NUM_SAMPLES] __attribute__((aligned(16))); 
static float features_buffer[SPECTROGRAM_SIZE] __attribute__((aligned(16)));

#ifdef ENABLE_WIFI_TELEMETRY
    static struct net_mgmt_event_callback wifi_cb;
    static struct net_mgmt_event_callback net_cb;
    static K_SEM_DEFINE(wifi_connected_sem, 0, 1);
    static K_SEM_DEFINE(ipv4_obtained_sem, 0, 1);
    static int telemetry_sock = -1;

    static void wifi_mgmt_event_handler(struct net_mgmt_event_callback *cb,
                                        uint32_t mgmt_event, struct net_if *iface)
    {
        const struct wifi_status *status = (const struct wifi_status *)cb->info;
        if (status->status) {
            printk("[WiFi] Connection Error: %d\n", status->status);
        } else {
            printk("[WiFi] Connected. Requesting DHCP...\n");
            k_sem_give(&wifi_connected_sem);
        }
    }

    static void net_mgmt_event_handler(struct net_mgmt_event_callback *cb,
                                       uint32_t mgmt_event, struct net_if *iface)
    {
        if (mgmt_event == NET_EVENT_IPV4_ADDR_ADD) {
            printk("[WiFi] IPv4 Address Obtained.\n");
            k_sem_give(&ipv4_obtained_sem);
        }
    }

    void setup_udp_socket(void) {
        telemetry_sock = zsock_socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (telemetry_sock < 0) {
            printk("[Telemetry] Failed to create socket\n");
        }
    }

    void send_telemetry_event(const char* msg) {
        if (telemetry_sock < 0) return;

        struct sockaddr_in dest_addr;
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(TELEMETRY_PORT);
        zsock_inet_pton(AF_INET, TELEMETRY_IP, &dest_addr.sin_addr);

        int ret = zsock_sendto(telemetry_sock, msg, strlen(msg), 0,
                               (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (ret < 0) {
            printk("[Telemetry] Send Failed\n");
        } else {
            printk("[Telemetry] Sent: %s\n", msg);
        }
    }

    void init_network_stack(void) {
        printk("Initializing Network Stack\n");
        net_mgmt_init_event_callback(&wifi_cb, wifi_mgmt_event_handler, NET_EVENT_WIFI_CONNECT_RESULT);
        net_mgmt_add_event_callback(&wifi_cb);
        net_mgmt_init_event_callback(&net_cb, net_mgmt_event_handler, NET_EVENT_IPV4_ADDR_ADD);
        net_mgmt_add_event_callback(&net_cb);

        struct net_if *iface = net_if_get_default();
        if (!iface) {
            printk("No network interface found\n");
            return;
        }

        struct wifi_connect_req_params params = {
            .ssid = WIFI_SSID,
            .ssid_length = strlen(WIFI_SSID),
            .psk = WIFI_PSK,
            .psk_length = strlen(WIFI_PSK),
            .security = WIFI_SECURITY_TYPE_PSK,
            .channel = WIFI_CHANNEL_ANY,
            .band = WIFI_FREQ_BAND_2_4_GHZ, 
            .mfp = WIFI_MFP_OPTIONAL
        };

        printk("Connecting to SSID: %s\n", WIFI_SSID);
        if (net_mgmt(NET_REQUEST_WIFI_CONNECT, iface, &params, sizeof(params))) {
            printk("Connection request failed\n");
            return;
        }

        if (k_sem_take(&wifi_connected_sem, K_SECONDS(20)) != 0) {
            printk("Connection timed out\n");
            return;
        }

        if (k_sem_take(&ipv4_obtained_sem, K_SECONDS(15)) != 0) {
            printk("DHCP timed out\n");
            return;
        }

        setup_udp_socket();
    }
#endif 

// digital injection, instead of microphone for more precise benchmarking and testing
void run_inference_test(const char* label, const int16_t* input_data, bool expect_target) {
    printk("\nTest: %s\n", label);

    if (input_data == NULL) {
        for(int i=0; i<NUM_SAMPLES; i++) audio_buffer[i] = (rand() % 10) - 5;
    } else {
        for(int i=0; i<NUM_SAMPLES; i++) audio_buffer[i] = input_data[i];
    }
    generate_spectrogram(audio_buffer, features_buffer);

    float peak_spectrogram = 0.0f;
    for(int i=0; i<SPECTROGRAM_SIZE; i++) {
        if(features_buffer[i] > peak_spectrogram) peak_spectrogram = features_buffer[i];
    }
    
    if (peak_spectrogram < SIGNAL_NOISE_FLOOR) {
        printk("Skipping Inference: Signal below noise floor (Max: %d)\n", (int)peak_spectrogram);
        return; 
    }


    gpio_pin_set_dt(&pin_debug, 1);
    uint32_t t_start = k_cycle_get_32();
    int confidence = run_inference(features_buffer);
    uint32_t t_end = k_cycle_get_32();
    
    gpio_pin_set_dt(&pin_debug, 0);

    uint32_t latency_ms = k_cyc_to_ms_ceil32(t_end - t_start);
    printk("Inference Time: %d ms | Confidence: %d%%\n", latency_ms, confidence);

    if (confidence > CONFIDENCE_THRESHOLD) {
        gpio_pin_set_dt(&led_target, 1);
        
        if (expect_target) {
            printk("Target Detected\n");
            #ifdef ENABLE_WIFI_TELEMETRY
            send_telemetry_event("KEYWORD_DETECTED_ON");
            #endif
        } else {
            printk("False Positive\n");
        }
    } else {
        gpio_pin_set_dt(&led_target, 0);
        if (expect_target) printk("False Negative\n");
        else printk("Correctly Ignored\n");
    }
}

int main(void)
{
    nrf_cache_enable(NRF_CACHE); // enabling instruction cache for performance
    if (!device_is_ready(led_target.port) || !device_is_ready(pin_debug.port)) return 0;
    gpio_pin_configure_dt(&led_target, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&pin_debug, GPIO_OUTPUT_ACTIVE);
    gpio_pin_set_dt(&led_target, 0);
    gpio_pin_set_dt(&pin_debug, 0);

    printk("*** nRF5340: Edge AI KWS System ***\n");
    if (inference_setup() != 0) {
        printk("AI Initialization Failed\n");
        return 0;
    }
    printk("AI Initialized\n");

    #ifdef ENABLE_WIFI_TELEMETRY
    init_network_stack();
    #endif

    while (1) { // low power idle state in the loop with the RTOS scheduler
        run_inference_test("Silence", NULL, false);
        k_msleep(2000);
        
        run_inference_test("Interference (Cat)", audio_cat2, false);
        k_msleep(2000);
        
        run_inference_test("Target (ON)", audio_on2, true);
        k_msleep(2000);
    }
    return 0;
}