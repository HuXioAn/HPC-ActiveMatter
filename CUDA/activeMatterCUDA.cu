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
                                    arrayPtr posX, arrayPtr posXTemp,
                                    arrayPtr posY, arrayPtr posYTemp,
                                    arrayPtr theta,arrayPtr thetaTemp );
__host__ int outputToFile(ofstream& outputFile, int birdNum, arrayPtr posX, arrayPtr posY, arrayPtr theta);

__host__ inline void swapPtr(arrayPtr &p1, arrayPtr &p2){
    arrayPtr temp = p1;
    p1 = p2;
    p2 = temp;
}

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
    arrayPtr posX(new arrayType[gPara.birdNum*(gPara.totalStep+1)]);
    arrayPtr posY(new arrayType[gPara.birdNum*(gPara.totalStep+1)]);
    arrayPtr theta(new arrayType[gPara.birdNum*(gPara.totalStep+1)]);

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
    if(cudaSuccess != cudaMalloc(&posXCuda, size*(gPara.totalStep+1))){
        printf("[!]Unable to alocate the posX Cuda mem.");
        exit(-1);
    }

    arrayPtr posYCuda;
    if(cudaSuccess != cudaMalloc(&posYCuda, size*(gPara.totalStep+1))){
        printf("[!]Unable to alocate the posY Cuda mem.");
        exit(-1);
    }

    arrayPtr thetaCuda;
    if(cudaSuccess != cudaMalloc(&thetaCuda, size*(gPara.totalStep+1))){
        printf("[!]Unable to alocate the theta Cuda mem.");
        exit(-1);
    }

    // arrayPtr thetaTempCuda;
    // if(cudaSuccess != cudaMalloc(&thetaTempCuda, size)){
    //     printf("[!]Unable to alocate the theta Cuda temp mem.");
    //     exit(-1);
    // }

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


    for(int step=0; step < gPara.totalStep; step++){

        computeActiveMatter<<<blockPerGrid, threadPerBlock>>>
            (gParaCuda, aParaCuda, 
            posXCuda+(step*gPara.birdNum), posXCuda+(step*gPara.birdNum+gPara.birdNum),
            posYCuda+(step*gPara.birdNum), posYCuda+(step*gPara.birdNum+gPara.birdNum),
            thetaCuda+(step*gPara.birdNum), thetaCuda+(step*gPara.birdNum+gPara.birdNum));
        
    }

    
    cudaDeviceSynchronize(); 
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[!]CUDA error:computeActiveMatter: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    if(OUTPUT_TO_FILE){
        ofstream outputFile;
        outputFile = ofstream(gPara.outputPath, ios::trunc);
        if(outputFile.is_open()){
            outputFile << std::fixed << std::setprecision(3);
        }else{
            cout << "[!]Unable to open output file: " << gPara.outputPath << endl;
            exit(-1);
        }

        //para
        outputFile << "generalParameter{" << "fieldLength=" << gPara.fieldLength << ",totalStep=" << gPara.totalStep << 
            ",birdNum=" << gPara.birdNum << "}" << endl;
        //data
        outputToFile(outputFile, gPara.birdNum, posX, posY, theta);

        cudaMemcpy(posX, posXCuda, size*(gPara.totalStep+1), cudaMemcpyDeviceToHost);
        cudaMemcpy(posY, posYCuda, size*(gPara.totalStep+1), cudaMemcpyDeviceToHost);
        cudaMemcpy(theta, thetaCuda, size*(gPara.totalStep+1), cudaMemcpyDeviceToHost);

        for(int step=1; step <= gPara.totalStep; step++){
            outputToFile(outputFile, gPara.birdNum, posX+(gPara.birdNum*step), posY+(gPara.birdNum*step), theta+(gPara.birdNum*step));
            //todo
        }
        
        outputFile.close();
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
                                    arrayPtr posX, arrayPtr posXTemp,
                                    arrayPtr posY, arrayPtr posYTemp,
                                    arrayPtr theta,arrayPtr thetaTemp ){

    auto bird = blockDim.x * blockIdx.x + threadIdx.x;

    if(bird >= gPara->birdNum)return;
    
    curandState state;
    curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    
    float observeRadiusSqr = powf(aPara->observeRadius,2);
    float inscribedSquareSideLengthHalf = aPara->observeRadius / sqrtf(2);


    //move
    posXTemp[bird] = posX[bird] + gPara->deltaTime * cosf(theta[bird]);
    posYTemp[bird] = posY[bird] + gPara->deltaTime * sinf(theta[bird]);

    //in the field
    posXTemp[bird] = fmod(posXTemp[bird]+gPara->fieldLength, gPara->fieldLength);
    posYTemp[bird] = fmod(posYTemp[bird]+gPara->fieldLength, gPara->fieldLength);

    
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

    __syncthreads();



}

__host__ int outputToFile(ofstream& outputFile, int birdNum, arrayPtr posX, arrayPtr posY, arrayPtr theta){
    //add current data to the file
    outputFile << "{" ;
    for(int bird=0; bird < birdNum; bird++){
        outputFile << posX[bird] << "," << posY[bird] << "," << theta[bird] << ";";
    }
    outputFile << "}" << endl;
    return 0;
}


