#!/bin/bash


cd /opt/external/opencv/
cmake -B build/arm -DCMAKE_TOOLCHAIN_FILE=./platforms/linux/arm-gnueabi.toolchain.cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ -DBUILD_LIST=tracking,imgcodecs,videoio,highgui,features2d,ml,xfeatures2d -DCMAKE_BUILD_TYPE=Release . 
cmake --build build/arm
cmake --install build/arm --prefix install/arm

cmake -B build/x64 -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules/ -DBUILD_LIST=tracking,imgcodecs,videoio,highgui,features2d,ml,xfeatures2d -DCMAKE_BUILD_TYPE=Release . 
cmake --build build/x64
cmake --install build/x64 --prefix install/x64

cd /opt/external/ncnn/
cmake -B build/arm -DCMAKE_TOOLCHAIN_FILE=./toolchains/arm-linux-gnueabihf.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON . 
cmake --build build/arm
cmake --install build/arm --prefix install/arm

cmake -B build/x64 -DNCNN_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_EXAMPLES=ON . 
cmake --build build/x64  
cmake --install build/x64 --prefix install/x64

cd /opt/external/zbar-0.10
env NM=nm CFLAGS="" ./configure --prefix=$(pwd)/build/arm --host=arm-none-linux-gnueabihf --build=x86_64-linux --enable-shared --without-gtk --without-python --without-qt --without-imagemagick --disable-video --without-xshm CC=arm-none-linux-gnueabihf-gcc CXX=arm-none-linux-gnueabihf-g++ 
make && make install

make clean \
   && env NM=nm CFLAGS="" ./configure --prefix=$(pwd)/build/x64 --enable-shared --without-gtk --without-python --without-qt --without-imagemagick --disable-video --without-xshm 
make && make install

