#include <Arduino.h>
#include <stdint.h>
#include <mlp_params.h>
#include <math.h>

struct qparams {
    float zero, scale;
};

struct qlayer {
    struct qparams input, weights, output;
};

// x must have 'cols' elements
void mvm(struct qlayer *qlayer, const int8_t *M, const int8_t* x, float *out, int rows, int cols) {
    int i, j;
    float mvm_scale;

    // TODO use python to export this value as a constant
    mvm_scale = (qlayer->input.scale * qlayer->weights.scale);

    for (i = 0; i < rows; i++) {
        float y;
        int16_t mult;
        int32_t acc = 0;

        for (j = 0; j < cols; j++) {
            // TODO: we can avoid this subtraction by "zero"
            mult = (M[i*cols + j] - qlayer->weights.zero) * (x[j] - qlayer->input.zero);
            acc += mult;
        }

        // TODO: we can avoid converting acc from int8 to float
        y = (mvm_scale * acc);
        y = qlayer->output.scale * (y - qlayer->output.zero);

        out[i] = y;
    }
}

void relu(float *x, float *out, int size) {
    int i;
    for (i = 0; i < size; i++)
        out[i] = x[i]>0?x[i]:0;
}

void quantize(float *x, struct qparams* qparams, int size, int8_t *out) {
    int i;

    for (i=0; i<size; i++) {
        out[i] = (int8_t)(round(x[i]/qparams->scale) + qparams->zero);
    }
}

int argmax(float *x, int size) {
    float max;
    int i, i_max;

    max = x[0];
    i_max = 0;

    for (i=1; i<size; i++) {
        if (x[i] > max) {
            max = x[i];
            i_max = i;
        }
    }

    return i_max;
}

void setup() {
    // TODO: allocate memory for layer inputs and outputs
    float layer1[96];
    float layer3[15];

    int8_t qlayer1[96];
    int8_t qlayer3[15];

    int subject_id;

    struct qlayer l1_qparams = {
        .input = {
            .zero = input_zero,
            .scale = input_scale
        },
        .weights = {
            .zero = layer_1_weights_zero,
            .scale = layer_1_weights_scale,
        },
        .output = {
            .zero = layer_1_zero,
            .scale = layer_1_scale,
        }
    };

    struct qlayer l3_qparams = {
        .input = {
            .zero = layer_1_zero,
            .scale = layer_1_scale
        },
        .weights = {
            .zero = layer_3_weights_zero,
            .scale = layer_3_weights_scale,
        },
        .output = {
            .zero = layer_3_zero,
            .scale = layer_3_scale,
        }
    };

    Serial.begin(921600);
    Serial.println("Starting MVM");

    mvm(&l1_qparams, layer_1_weights, input, layer1, 96, 165);
    relu(layer1, layer1, 96);

    // quantize output
    quantize(layer1, &l1_qparams.output, 96, qlayer1);

    mvm(&l3_qparams, layer_3_weights, qlayer1, layer3, 15, 96);

    subject_id = argmax(layer3, 15);

    Serial.printf("Finished MVM, subject: %d\n", subject_id);
}

void loop() {

}
