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
#include <algorithm>

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

__global__ void computeActiveMatter(generalPara_t* gPara, activePara_t* aPara, 
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
    if(cudaSuccess != cudaMalloc(&posXCuda, size * 4)){
        printf("[!]Unable to alocate the Cuda mem.");
        exit(-1);
    }

    arrayPtr posYCuda = posXCuda + gPara.birdNum;
    arrayPtr thetaCuda = posXCuda + gPara.birdNum * 2;
    arrayPtr thetaTempCuda = posXCuda + gPara.birdNum * 3;


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

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[!]CUDA error:before: %s\n", cudaGetErrorString(error));
        exit(1);
    }


//**********************************************************
//kernel
    // 256 threads per block
    auto threadPerBlock = 256;
    auto blockPerGrid = (gPara.birdNum + threadPerBlock - 1) / threadPerBlock;

    
    cudaDeviceProp prop;
    int currentDevice = 0;
    cudaGetDevice(&currentDevice);
    cudaGetDeviceProperties(&prop, currentDevice);
    
    size_t l2Size = min(int(prop.l2CacheSize * 0.25), prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2Size);

    size_t window_size = min((unsigned long)prop.accessPolicyMaxWindowSize, (unsigned long)(size * 4));                        // Select minimum of user defined num_bytes and max window size.

    cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(posXCuda);               // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
    stream_attribute.accessPolicyWindow.hitRatio  = 0.8;                                        // Hint for cache hit ratio
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming; 

    cout << "[*] CUDA L2 persisting L2 size: " << l2Size << ", Window size: " << window_size << endl;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("[!]CUDA error:L2: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    using namespace std::chrono;
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();


    // output of the params anf the first step
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
        //data
        //outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
    }


    for(int step=0; step < gPara.totalStep; step++){

        computeActiveMatter<<<blockPerGrid, threadPerBlock>>>
            (gParaCuda, aParaCuda, 
            posXCuda, 
            posYCuda, 
            thetaCuda, thetaTempCuda);
        

        //dual-buffer, swap ptr
        swapPtr(thetaCuda, thetaTempCuda);

        //making memcpy and computing parallel does not make things better
        //for the small amount of data, synchronization brings relatively large overhead
        if(OUTPUT_TO_FILE){
            //write previous step to output
            //making output io and computing parallel makes big difference
            outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
            cudaDeviceSynchronize(); 

            cudaMemcpy(posX, posXCuda, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(posY, posYCuda, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(theta, thetaCuda, size, cudaMemcpyDeviceToHost);

        }
        

    }
    if(OUTPUT_TO_FILE){outputToFile(outputFile, gPara.birdNum, posX, posY, theta);} // last step
    
    cudaDeviceSynchronize(); 

    if(OUTPUT_TO_FILE)outputFile.close();

    error = cudaGetLastError();
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
__global__ void computeActiveMatter(generalPara_t* gPara, activePara_t* aPara, 
                                    arrayPtr posX, 
                                    arrayPtr posY, 
                                    arrayPtr theta,arrayPtr thetaTemp ){

    auto bird = blockDim.x * blockIdx.x + threadIdx.x;

    if(bird >= gPara->birdNum)return;
    
    curandState state;
    curand_init(clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);
    
    float observeRadiusSqr = powf(aPara->observeRadius,2);
    float inscribedSquareSideLengthHalf = aPara->observeRadius / sqrtf(2);


    //move
    posX[bird] = posX[bird] + gPara->deltaTime * cosf(theta[bird]);
    posY[bird] = posY[bird] + gPara->deltaTime * sinf(theta[bird]);

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

    __syncthreads();



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


