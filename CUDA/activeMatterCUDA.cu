/*
cuda ActiveMatter simulation
Auther: Anton
Create Time: 22/04/24

*/

#include <iostream>
#include <cmath>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <cuda.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"


using namespace std;

constexpr int DEFAULT_BIRD_NUM = 500; 
constexpr bool OUTPUT_TO_FILE = true;

typedef struct generalPara_s
{
    float fieldLength;
    float deltaTime;
    int totalStep;
    int birdNum;

    int randomSeed;
    string outputPath;

}generalPara_t;

typedef struct activePara_s
{
    float velocity;
    float fluctuation; //in radians
    float observeRadius;

}activePara_t;

using arrayType = float;
using arrayPtr = arrayType*;

//0-1 float random
mt19937 randomGen;
uniform_real_distribution<float> randomDist;

__global__ void computeActiveMatter(generalPara_t* gPara, activePara_t* aPara, 
                                    arrayPtr posX, arrayPtr posY, arrayPtr theta, arrayPtr thetaTemp);

__host__ int main(int argc, char* argv[]){

    auto birdNum = DEFAULT_BIRD_NUM;
    if(argc > 1){
        birdNum = atoi(argv[1]);
        if(birdNum == 0)birdNum = DEFAULT_BIRD_NUM;
    }

    //load the params
    generalPara_s gPara = {
        .fieldLength = 10.0,
        .deltaTime = 0.2,
        .totalStep = 500,
        .birdNum = birdNum,
        .randomSeed = static_cast<int>(time(nullptr)),
        .outputPath = "./output.plot",
    };

    activePara_s aPara = {
        .velocity = 1.0,
        .fluctuation = 0.5,
        .observeRadius = 1.0,
    };

    randomGen = mt19937(gPara.randomSeed);
    randomDist = uniform_real_distribution<arrayType>(0,1);

//*******************************************************
//initialize the host mem

    //initialize the data
    arrayPtr posX(new arrayType[gPara.birdNum]);
    arrayPtr posY(new arrayType[gPara.birdNum]);
    arrayPtr theta(new arrayType[gPara.birdNum]);

    for(int i=0; i < gPara.birdNum; i++){
        //randomize the pos and theta
        auto randomFloat = randomDist(randomGen);
        posX[i] = randomFloat * gPara.fieldLength;

        randomFloat = randomDist(randomGen);
        posY[i] = randomFloat * gPara.fieldLength;

        randomFloat = randomDist(randomGen);
        theta[i] = randomFloat * M_PI * 2;
    }

//**********************************************************
//initialize the device mem
    auto size = gPara.birdNum * sizeof(arrayType);

    arrayPtr posXCuda;
    if(cudaSuccess != cudaMalloc(&posXCuda, size)){
        printf("[!]Unable to alocate the posX Cuda mem.");
        exit(-1);
    }

    arrayPtr posYCuda;
    if(cudaSuccess != cudaMalloc(&posYCuda, size)){
        printf("[!]Unable to alocate the posY Cuda mem.");
        exit(-1);
    }

    arrayPtr thetaCuda;
    if(cudaSuccess != cudaMalloc(&thetaCuda, size)){
        printf("[!]Unable to alocate the theta Cuda mem.");
        exit(-1);
    }

    arrayPtr thetaTempCuda;
    if(cudaSuccess != cudaMalloc(&thetaTempCuda, size)){
        printf("[!]Unable to alocate the theta Cuda temp mem.");
        exit(-1);
    }

    generalPara_s* gParaCuda;
    if(cudaSuccess != cudaMalloc(&gParaCuda, sizeof(generalPara_s))){
        printf("[!]Unable to alocate the gPara Cuda mem.");
        exit(-1);
    }

    activePara_s* aParaCuda;
    if(cudaSuccess != cudaMalloc(&aParaCuda, sizeof(activePara_s))){
        printf("[!]Unable to alocate the aPara Cuda mem.");
        exit(-1);
    }

    cudaMemcpy(posXCuda, posX, size, cudaMemcpyHostToDevice);
    cudaMemcpy(posYCuda, posY, size, cudaMemcpyHostToDevice);
    cudaMemcpy(thetaCuda, theta, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gParaCuda, &gPara, sizeof(generalPara_s), cudaMemcpyHostToDevice);
    cudaMemcpy(aParaCuda, &aPara, sizeof(activePara_s), cudaMemcpyHostToDevice);


//**********************************************************
//kernel

    auto threadPerBlock = 256;
    auto blockPerGrid = (gPara.birdNum + threadPerBlock - 1) / threadPerBlock;


    using namespace std::chrono;
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();

    computeActiveMatter<<<blockPerGrid, threadPerBlock>>>(gParaCuda, aParaCuda, posXCuda, posYCuda, thetaCuda, thetaTempCuda);
    //output


    cudaDeviceSynchronize();    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[!]CUDA error:computeActiveMatter: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    cout << "Compute Time: " << duration << "ms" << endl;


//***********************************************************

    delete[] posX;
    delete[] posY;
    delete[] theta;

    return 0;
    
}


__global__ void computeActiveMatter(generalPara_t* gPara, activePara_t* aPara, 
                                    arrayPtr posX, arrayPtr posY, arrayPtr theta,arrayPtr thetaTemp ){

    auto bird = blockDim.x * blockIdx.x + threadIdx.x;

    if(bird >= gPara->birdNum)return;
    
    curandState state;
    curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    
    float observeRadiusSqr = powf(aPara->observeRadius,2);
    float inscribedSquareSideLengthHalf = aPara->observeRadius / sqrtf(2);

    for(int step=0; step < gPara->totalStep; step++){ //steps

        //move
        posX[bird] += gPara->deltaTime * cosf(theta[bird]);
        posY[bird] += gPara->deltaTime * sinf(theta[bird]);

        //in the field
        posX[bird] = fmod(posX[bird]+gPara->fieldLength, gPara->fieldLength);
        posY[bird] = fmod(posY[bird]+gPara->fieldLength, gPara->fieldLength);

        
        //adjust theta
        float sx = 0,sy = 0; 

        for(int oBird=0; oBird < gPara->birdNum; oBird++){ //observe other birds, self included

            auto xDiffAbs = fabsf(posX[bird]-posX[oBird]);
            auto yDiffAbs = fabsf(posY[bird]-posY[oBird]);
            
            if((xDiffAbs > aPara->observeRadius) || 
                (yDiffAbs > aPara->observeRadius) 
                || ((xDiffAbs > inscribedSquareSideLengthHalf) && (yDiffAbs > inscribedSquareSideLengthHalf))
                )continue;//ignore birds outside the circumscribed square and 4 corners

            if((xDiffAbs < inscribedSquareSideLengthHalf) && 
                (yDiffAbs < inscribedSquareSideLengthHalf)){ //birds inside the inscribed square
                sx += cosf(theta[oBird]);
                sy += sinf(theta[oBird]);
            }else{
                auto distPow2 = powf(xDiffAbs, 2) + powf(yDiffAbs, 2);
                if(distPow2 < observeRadiusSqr){ //observed
                    sx += cosf(theta[oBird]);
                    sy += sinf(theta[oBird]);
                }
            }
        }
        thetaTemp[bird] = atan2f(sy, sx) + (curand_uniform(&state) - 0.5) * aPara->fluctuation; //new theta
    

        //dual-buffer, swap ptr
        auto tempPtr = theta;
        theta = thetaTemp;
        thetaTemp = tempPtr;

        
    }


}

__host__ int outputToFile(generalPara_t gPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta);


