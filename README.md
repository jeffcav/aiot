# Experiments on Artificial Inteligence of Things (AIOT)

## Instructions

First, create a python environment and download dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, open [aiot.ipynb](aiot.ipynb) and run all cells except the last ones, that communicate with an ESP-32 powered device.

Compile and deploy the neutral network on your ESP32-S3 device using platformIO.
The project is under [src/embedded/esp32s3](src/embedded/esp32s3)

Finally, run the last cell in [aiot.ipynb](aiot.ipynb) to send inputs to the ESP32-S3 device and run inferences.

## Sidenotes

1. If you have platformIO installed and configured for the project under [esp32s3](src/embedded/esp32s3), you can inspect object files with:

```bash
~/.platformio/packages/toolchain-xtensa-esp32s3/bin/xtensa-esp32s3-elf-objdump -d src/embedded/esp32s3/.pio/build/nodemcu-32s2/src/main.cpp.o
```

## References

[1] [Benjamin Fuhrer's material](https://github.com/benja263/Integer-Only-Inference-for-Deep-Learning-in-Native-C)

[2] [PTQ to 8-bits with Pytorch](https://karanbirchahal.medium.com/how-to-quantise-an-mnist-network-to-8-bits-in-pytorch-no-retraining-required-from-scratch-39f634ac8459)

[3] [Pytorch's min-max quantization implmentation](https://github.com/pytorch/pytorch/blob/d542aab5c1bc544f9dc0eb5632bfe4432223d890/test/fx/quantization.py)

[4] [Background on fixed-point matrix multiplication](https://github.com/google/gemmlowp/blob/master/doc/quantization.md#implementation-of-quantized-matrix-multiplication)
