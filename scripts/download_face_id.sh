#!/bin/bash
# Download the ArcFace (IR-SE50) checkpoint used by the FaceID guidance loss.

wget "https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0" -O src/losses/checkpoints/model_ir_se50.pth
