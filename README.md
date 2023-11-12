# M5 UnitV2 Framework

## Build example

```sh
$ cmake -B build/arm -DTARGET_COMPILE=arm -DTARGET=camera_capture .
$ cmake --build build/arm

$ cmake -B build/x64 -DTARGET_COMPILE=x64 -DTARGET=camera_capture .
$ cmake --build build/x64
```


## Toolchain

gcc-arm-10.2-2020.11-x86_64-arm-none-linux-gnueabihf.tar.xz

[@download page](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads)

[@download link](https://developer.arm.com/-/media/Files/downloads/gnu-a/10.2-2020.11/binrel/gcc-arm-10.2-2020.11-x86_64-arm-none-linux-gnueabihf.tar.xz?revision=d0b90559-3960-4e4b-9297-7ddbc3e52783&la=en&hash=985078B758BC782BC338DB947347107FBCF8EF6B)

```sh
$ mkdir ~/external
$ cd ~/external
$ curl -LO https://developer.arm.com/-/media/Files/downloads/gnu-a/10.2-2020.11/binrel/gcc-arm-10.2-2020.11-x86_64-arm-none-linux-gnueabihf.tar.xz
$ tar Jxfv gcc-arm-10.2-2020.11-x86_64-arm-none-linux-gnueabihf.tar.xz

```


## Dependent library

OpenCV  4.4.0  +  OpenCV's extra modules   4.4.0

[@opencv](https://github.com/opencv/opencv)

[@opencv_contrib](https://github.com/opencv/opencv_contrib)

```sh
$ cd ~/external/
$ git clone https://github.com/opencv/opencv.git -b 4.4.0 --depth 1 
$ git clone https://github.com/opencv/opencv_contrib.git -b 4.4.0 --depth 1

cmake -B build/arm -DCMAKE_TOOLCHAIN_FILE=./platforms/linux/arm-gnueabi.toolchain.cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ -DBUILD_LIST=tracking,imgcodecs,videoio,highgui,features2d,ml,xfeatures2d -DCMAKE_BUILD_TYPE=Release . 
cmake --build build/arm
cmake --install build/arm --prefix install/arm

cmake -B build/x64 -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ -DBUILD_LIST=tracking,imgcodecs,videoio,highgui,features2d,ml,xfeatures2d -DCMAKE_BUILD_TYPE=Release . 
cmake --build build/x64
cmake --install build/x64 --prefix install/x64
```

NCNN

[@ncnn](https://github.com/Tencent/ncnn)

```sh
$ cd ~/external/
$ git clone https://github.com/Tencent/ncnn.git -b 20231027 --depth 1
$ cmake -B build/arm -DCMAKE_TOOLCHAIN_FILE=./toolchains/arm-linux-gnueabihf.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON . 
$ cmake --build build/arm
$ cmake --install build/arm --prefix install/arm

$ cmake -B build/x64 -DNCNN_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_EXAMPLES=ON . 
$ cmake --build build/x64  
$ cmake --install build/x64 --prefix install/x64
```

ZBAR

[@ZBar](https://github.com/ZBar/ZBar)

```sh
$ cd ~/external/
$ curl -LO https://jaist.dl.sourceforge.net/project/zbar/zbar/0.10/zbar-0.10.tar.bz2 
$ tar -jxvf zbar-0.10.tar.bz2

$ cd ~/external/zbar-0.10
$ env NM=nm CFLAGS="" ./configure --prefix=$(pwd)/build/arm --host=arm-none-linux-gnueabihf --build=x86_64-linux --enable-shared --without-gtk --without-python --without-qt --without-imagemagick --disable-video --without-xshm CC=arm-none-linux-gnueabihf-gcc CXX=arm-none-linux-gnueabihf-g++ 
$ make && make install

$ make clean \
    && env NM=nm CFLAGS="" ./configure --prefix=$(pwd)/build/x64 --enable-shared --without-gtk --without-python --without-qt --without-imagemagick --disable-video --without-xshm 
$ make && make install
```
