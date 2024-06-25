/**
 * @file activeMatterCUDA.cpp
 * @brief accelerated activeMatter simulation on CUDA GPU
 * @details CUDA accelerated code, L2-persisting disabled
 * @author Andong Hu
 * @date 2024-4-22
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

//! default bird number if not given
constexpr int DEFAULT_BIRD_NUM = 500; 

//! computing result output control
constexpr bool OUTPUT_TO_FILE = true;

/**
 * @brief structure of general parameters in the simulation
 * @details contains the parameters of the simulation, 
 * such as field length, step number, random seed...
*/
typedef struct generalPara_s
{
    float fieldLength;  ///< side length of the square simulation field
    float deltaTime;    ///< the time between steps, to control the movement
    int totalStep;      ///< steps number of the simulation
    int birdNum;        ///< bird number in the simulation

    int randomSeed;     ///< seed for the random generator
    string outputPath;  ///< path for the output file

}generalPara_t;

/**
 * @brief structure of parameters of birds in the simulation
 * @details parameters like the velocity of movement, 
 * index of fluctuation in orientation and the radius of observed area
*/
typedef struct activePara_s
{
    float velocity;     ///< movement speed of birds
    float fluctuation;  ///< index of fluctuation in theta adjustment, in radian
    float observeRadius;///< radius of the observed area

}activePara_t;

using arrayType = float;
//! alias for the data type pointer
using arrayPtr = arrayType*;

//! 0-1 float random number generator
mt19937 randomGen;
uniform_real_distribution<float> randomDist;


__global__ void move(generalPara_t* gPara, activePara_t* aPara, 
                                    arrayPtr posX, 
                                    arrayPtr posY, 
                                    arrayPtr theta);

__global__ void adjust(generalPara_t* gPara, activePara_t* aPara, 
                                    arrayPtr posX, 
                                    arrayPtr posY, 
                                    arrayPtr theta,arrayPtr thetaTemp );
__host__ int outputToFile(ofstream& outputFile, int birdNum, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta);


/**
 * @brief host funtion swap two pointers
 * 
 * @param[in,out] p1 the first pointer
 * @param[in,out] p2 the second pointer
*/
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
    auto size = gPara.birdNum * sizeof(arrayType);
    //initialize the data

    arrayPtr posX, posXNext;
    arrayPtr posY, posYNext;
    arrayPtr theta, thetaNext;
    { // fold me
        if(cudaSuccess != cudaHostAlloc(&posX, size, cudaHostAllocDefault)){
            printf("[!]Unable to alocate the posX Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaHostAlloc(&posY, size, cudaHostAllocDefault)){
            printf("[!]Unable to alocate the posY Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaHostAlloc(&theta, size, cudaHostAllocDefault)){
            printf("[!]Unable to alocate the theta Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaHostAlloc(&posXNext, size, cudaHostAllocDefault)){
            printf("[!]Unable to alocate the posXNext Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaHostAlloc(&posYNext, size, cudaHostAllocDefault)){
            printf("[!]Unable to alocate the posYNext Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaHostAlloc(&thetaNext, size, cudaHostAllocDefault)){
            printf("[!]Unable to alocate the thetaNext Cuda mem.");
            exit(-1);
        }
    }
    

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

    arrayPtr posXCuda;
    arrayPtr posYCuda;
    arrayPtr thetaCuda;
    arrayPtr thetaTempCuda;
    generalPara_s* gParaCuda;
    activePara_s* aParaCuda;
    { // fold me 
        if(cudaSuccess != cudaMalloc(&posXCuda, size)){
            printf("[!]Unable to alocate the posX Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaMalloc(&posYCuda, size)){
            printf("[!]Unable to alocate the posY Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaMalloc(&thetaCuda, size)){
            printf("[!]Unable to alocate the theta Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaMalloc(&thetaTempCuda, size)){
            printf("[!]Unable to alocate the theta Cuda temp mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaMalloc(&gParaCuda, sizeof(generalPara_s))){
            printf("[!]Unable to alocate the gPara Cuda mem.");
            exit(-1);
        }

        if(cudaSuccess != cudaMalloc(&aParaCuda, sizeof(activePara_s))){
            printf("[!]Unable to alocate the aPara Cuda mem.");
            exit(-1);
        }
    }
    

    cudaMemcpy(posXCuda, posX, size, cudaMemcpyHostToDevice);
    cudaMemcpy(posYCuda, posY, size, cudaMemcpyHostToDevice);
    cudaMemcpy(thetaCuda, theta, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gParaCuda, &gPara, sizeof(generalPara_s), cudaMemcpyHostToDevice);
    cudaMemcpy(aParaCuda, &aPara, sizeof(activePara_s), cudaMemcpyHostToDevice);


//**********************************************************
//kernel
    // 256 threads per block
    auto threadPerBlock = 256;
    auto blockPerGrid = (gPara.birdNum + threadPerBlock - 1) / threadPerBlock;

    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i)cudaStreamCreate(&stream[i]);

    cudaEvent_t event[2];
    // move to adjust
    cudaEventCreateWithFlags(&event[0], cudaEventDefault | cudaEventDisableTiming);
    // adjust to move
    cudaEventCreateWithFlags(&event[1], cudaEventDefault | cudaEventDisableTiming);
    // for host output
    cudaEvent_t outputEvent[gPara.totalStep];
    for (int i = 0; i < gPara.totalStep; ++i)
    cudaEventCreateWithFlags(&outputEvent[i], cudaEventDefault | cudaEventDisableTiming | cudaEventBlockingSync);

    using namespace std::chrono;
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();


    // output of the params 
    ofstream outputFile;
    if(OUTPUT_TO_FILE){//save the parameter,first step to file
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

        outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
    }

    for(int step=0; step < gPara.totalStep; step++){
        { // stream 1

            cudaStreamWaitEvent(stream[0], event[1]);

            move<<<blockPerGrid, threadPerBlock, 0, stream[0]>>>
                (gParaCuda, aParaCuda, 
                posXCuda, 
                posYCuda, 
                thetaCuda);
            
            cudaEventRecord(event[0], stream[0]);

            //copy to pos buf
            cudaMemcpyAsync(posXNext, posXCuda, size, cudaMemcpyDeviceToHost, stream[0]);
            cudaMemcpyAsync(posYNext, posYCuda, size, cudaMemcpyDeviceToHost, stream[0]);
        }


        { // stream 2
            
            cudaStreamWaitEvent(stream[1], event[0]);

            adjust<<<blockPerGrid, threadPerBlock, 0, stream[1]>>>
                (gParaCuda, aParaCuda, 
                posXCuda, 
                posYCuda, 
                thetaCuda, thetaTempCuda);

            cudaEventRecord(event[1], stream[1]);

            cudaMemcpyAsync(thetaNext, thetaCuda, size, cudaMemcpyDeviceToHost, stream[1]);

            cudaEventRecord(outputEvent[step], stream[1]);
            
        }
        
        //dual-buffer, swap ptr
        swapPtr(thetaCuda, thetaTempCuda);
        swapPtr(posX, posXNext);
        swapPtr(posY, posYNext);
        swapPtr(theta, thetaNext);
   
    }

    if(OUTPUT_TO_FILE){ // output on host
        if((gPara.totalStep % 2) != 0){
            swapPtr(posX, posXNext);
            swapPtr(posY, posYNext);
            swapPtr(theta, thetaNext);
        }



        for(int step=0; step < gPara.totalStep; step++){

            swapPtr(posX, posXNext);
            swapPtr(posY, posYNext);
            swapPtr(theta, thetaNext);

            cudaEventSynchronize(outputEvent[step]);
            outputToFile(outputFile, gPara.birdNum, posX, posY, theta);

        }

        outputFile.close();
    }
    
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

    cudaFreeHost(posX);
    cudaFreeHost(posY);
    cudaFreeHost(theta);
    cudaFreeHost(posXNext);
    cudaFreeHost(posYNext);
    cudaFreeHost(thetaNext);

    return 0;
    
}


__global__ void move(generalPara_t* gPara, activePara_t* aPara, 
                                    arrayPtr posX, 
                                    arrayPtr posY, 
                                    arrayPtr theta){

    auto bird = blockDim.x * blockIdx.x + threadIdx.x;

    if(bird >= gPara->birdNum)return;

    //move
    posX[bird] = posX[bird] + gPara->deltaTime * cosf(theta[bird]);
    posY[bird] = posY[bird] + gPara->deltaTime * sinf(theta[bird]);

    //in the field
    posX[bird] = fmod(posX[bird]+gPara->fieldLength, gPara->fieldLength);
    posY[bird] = fmod(posY[bird]+gPara->fieldLength, gPara->fieldLength);

}

/**
 * @brief one step computation of the activeMatter
 * 
 * @details compute position(posX, posY) and orientation(theta) onr step, 
 * 
 * @param[in] gPara structure of general paramters 
 * @param[in] aPara structure of bird parameters
 * @param[in,out] posX pointer to the array of birds' position X
 * @param[in,out] posY pointer to the array of birds' position Y
 * @param[in] theta pointer to the array of birds' theta
 * @param[out] thetaTemp pointer to the temporary array of birds' theta
*/
__global__ void adjust(generalPara_t* gPara, activePara_t* aPara, 
                                    arrayPtr posX, 
                                    arrayPtr posY, 
                                    arrayPtr theta,arrayPtr thetaTemp ){

    auto bird = blockDim.x * blockIdx.x + threadIdx.x;

    if(bird >= gPara->birdNum)return;
    
    curandState state;
    curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    
    float observeRadiusSqr = powf(aPara->observeRadius,2);
    float inscribedSquareSideLengthHalf = aPara->observeRadius / sqrtf(2);
    
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

}

/**
 * @brief write data of one step to file
 * 
 * @param[in] outputFile reference of opened file stream of the output file
 * @param[in] birdNum number of the birds in the three arrays
 * @param[in] posX reference of the pointer to the array of birds' position X
 * @param[in] posY reference of the pointer to the array of birds' position Y
 * @param[in] theta reference of the pointer to the array of birds' theta
*/
__host__ int outputToFile(ofstream& outputFile, int birdNum, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta){
    //add current data to the file
    outputFile << "{" ;
    for(int bird=0; bird < birdNum; bird++){
        outputFile << posX[bird] << "," << posY[bird] << "," << theta[bird] << ";";
    }
    outputFile << "}" << endl;
    return 0;
}


