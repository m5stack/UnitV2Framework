# M5 UnitV2 Framework

## Toolchain

gcc-arm-10.2-2020.11-x86_64-arm-none-linux-gnueabihf.tar.xz

[@download page](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads)

[@download link](https://developer.arm.com/-/media/Files/downloads/gnu-a/10.2-2020.11/binrel/gcc-arm-10.2-2020.11-x86_64-arm-none-linux-gnueabihf.tar.xz?revision=d0b90559-3960-4e4b-9297-7ddbc3e52783&la=en&hash=985078B758BC782BC338DB947347107FBCF8EF6B)


## Dependent library

OpenCV  4.4.0  +  OpenCV's extra modules   4.4.0

[@opencv](https://github.com/opencv/opencv)

[@opencv_contrib](https://github.com/opencv/opencv_contrib)

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=../arm-gnueabi.toolchain.cmake -DOPENCV_EXTRA_MODULES_PATH=<PATH TO opencv_contrib/modules> -DBUILD_LIST=tracking,imgcodecs,videoio,highgui,features2d,ml,xfeatures2d -DCMAKE_BUILD_TYPE=Release ../../..
```

NCNN

[@ncnn](https://github.com/Tencent/ncnn)

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON ..
```

ZBAR

[@ZBar](https://github.com/ZBar/ZBar)

```sh
./configure --prefix=$(pwd)/build --host=arm-none-linux-gnueabihf --enable-shared --without-gtk --without-python --without-qt --without-imagemagick --disable-video CC=arm-none-linux-gnueabihf-gcc CXX=arm-none-linux-gnueabihf-g++
```
