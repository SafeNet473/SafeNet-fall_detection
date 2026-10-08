#include <stdio.h>
#include <stdint.h>

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>

#ifdef CONFIG_LSM6DSO_TRIGGER

static uint32_t sample_count = 0;
static int64_t prev_us = -1;

static void trigger_handler(
    const struct device *dev,
    const struct sensor_trigger *trig)
{
    struct sensor_value accel[3];

    // Callback execution time, not exact IMU hardware timestamp
    int64_t now_us =
        k_ticks_to_us_floor64(k_uptime_ticks());

    int64_t dt_us = 0;

    if (prev_us >= 0) {
        dt_us = now_us - prev_us;
    }
    prev_us = now_us;

    int ret = sensor_sample_fetch_chan(
        dev, SENSOR_CHAN_ACCEL_XYZ);

    if (ret != 0) {
        return;
    }

    ret = sensor_channel_get(
        dev, SENSOR_CHAN_ACCEL_XYZ, accel);

    if (ret != 0) {
        return;
    }

    float ax =
        sensor_value_to_float(&accel[0]) / 9.80665f;
    float ay =
        sensor_value_to_float(&accel[1]) / 9.80665f;
    float az =
        sensor_value_to_float(&accel[2]) / 9.80665f;

    sample_count++;

    // Debug only: print much less frequently
    if (sample_count % 208 == 0) {
        printf("Samples: %u, dt: %lld us, "
               "accel: %.3f %.3f %.3f g\n",
               sample_count,
               (long long)dt_us,
               (double)ax,
               (double)ay,
               (double)az);
    }
}

int main(void)
{
    const struct device *dev =
        DEVICE_DT_GET_ONE(st_lsm6dso);

    if (!device_is_ready(dev)) {
        printf("IMU not ready\n");
        return 0;
    }

    struct sensor_value odr = {
        .val1 = 208,
        .val2 = 0
    };

    int ret = sensor_attr_set(
        dev,
        SENSOR_CHAN_ACCEL_XYZ,
        SENSOR_ATTR_SAMPLING_FREQUENCY,
        &odr
    );

    printf("ODR configure ret = %d\n", ret);

    if (ret != 0) {
        return 0;
    }

    static const struct sensor_trigger trig = {
        .type = SENSOR_TRIG_DATA_READY,
        .chan = SENSOR_CHAN_ACCEL_XYZ
    };

    ret = sensor_trigger_set(
        dev, &trig, trigger_handler);

    printf("Trigger configure ret = %d\n", ret);

    if (ret != 0) {
        return 0;
    }

    printf("IMU trigger enabled\n");

    return 0;
}

#else
#error "LSM6DSO trigger support is not enabled"
#endif