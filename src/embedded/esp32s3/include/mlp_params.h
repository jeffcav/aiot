#ifndef MLP_PARAMS
#define MLP_PARAMS

#include <stdint.h>

extern const int8_t input[165];
extern const float input_zero;
extern const float input_scale;

extern const int8_t layer_1_weights[15840];
extern const float layer_1_weights_zero;
extern const float layer_1_weights_scale;

extern const float layer_1_zero;
extern const float layer_1_scale;

extern const int8_t layer_3_weights[1440];
extern const float layer_3_weights_zero;
extern const float layer_3_weights_scale;

extern const float layer_3_zero;
extern const float layer_3_scale;


#endif // end of MLP_PARAMS
