# cuda-nlmeans

对比了不同的经典图像去噪方法的效果和 Non-local Means 去噪方法的加速效果。结果请查看 `article.pdf`。

## 去噪方法和加速平台

去噪方法：中值滤波，高斯滤波，双边滤波和非局部均值滤波。

加速平台：单线程，多线程(OpenMP)，CUDA。

## 在 Google Colab 中运行

~~使用 [vcpkg](https://github.com/microsoft/vcpkg) 搭建了环境。~~

使用 apt 安装了 opencv-contrib。其中，CUDA 是可选项。如果需要运行 CUDA 相关的代码，请修改 Colab 的运行时为**硬件加速器-GPU**。

Google Colab 的运行脚本：

```
# cuda-nlmeans env setup
!apt update
# !apt install -y build-essential tar curl zip unzip gcc g++ gdb make cmake bison
# !apt install -y autoconf libsass-dev libtool libxrandr-dev libxi-dev
# !apt install -y libxcursor-dev libxinerama-dev
# !git clone https://github.com/microsoft/vcpkg
# !./vcpkg/bootstrap-vcpkg.sh
# %cd /content/vcpkg
# !./vcpkg install opencv3[contrib]:x64-linux
!apt install -y libopencv-contrib-dev

# cuda-nlmeans build and test
%cd /content
!git clone https://github.com/zixgo/cuda-nlmeans.git
%cd /content/cuda-nlmeans
!cmake -DCMAKE_BUILD_TYPE=Release -S. -Bout "-DCMAKE_TOOLCHAIN_FILE=/content/vcpkg/scripts/buildsystems/vcpkg.cmake"
!cmake --build out
!./cuda-nlmeans
```

## 其他

如果有关于代码的疑问，欢迎提 issue。If any questions, an issue is welcomed.
