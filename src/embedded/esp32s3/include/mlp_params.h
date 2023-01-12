#ifndef MLP_PARAMS
#define MLP_PARAMS

#include <stdint.h>

extern const int8_t input[165];
extern const int8_t output;
extern const int8_t input_zero;
extern const float input_scale;

#define layer_1_fpmult 0.7873240920947224
extern const int32_t layer_1_multiplier;
extern const int32_t layer_1_rshifts;
extern const int8_t layer_1_weights[15840];
extern const int8_t layer_1_weights_zero;
extern const float layer_1_weights_scale;

extern const int8_t layer_1_zero;
extern const float layer_1_scale;

#define layer_3_fpmult 0.9818835089457887
extern const int32_t layer_3_multiplier;
extern const int32_t layer_3_rshifts;
extern const int8_t layer_3_weights[1440];
extern const int8_t layer_3_weights_zero;
extern const float layer_3_weights_scale;

extern const int8_t layer_3_zero;
extern const float layer_3_scale;


#endif // end of MLP_PARAMS
