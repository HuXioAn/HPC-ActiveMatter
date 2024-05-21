/**
 * @file activeMatter_rawPtr.cpp
 * @brief Optimized serial code for activeMatter
 * @details Single-threaded cpp code for activeMatter, all optimization applied
 * @author Andong Hu
 * @date 2024-5-21
 */


#include <iostream>
#include <cmath>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <chrono>

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


//! alias for the data type pointer
using arrayPtr = float*;

//! 0-1 float random number generator
mt19937 randomGen;
uniform_real_distribution<float> randomDist;

void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta);



int main(int argc, char* argv[]){

    auto birdNum = DEFAULT_BIRD_NUM;
    if(argc > 1){
        birdNum = atoi(argv[1]);
        if(birdNum == 0)birdNum = DEFAULT_BIRD_NUM;
    }

    // the default general parameters
    generalPara_s gPara = {
        .fieldLength = 10.0,
        .deltaTime = 0.2,
        .totalStep = 500,
        .birdNum = birdNum,
        .randomSeed = static_cast<int>(time(nullptr)),
        .outputPath = "./output.plot",
    };

    // the default bird parameters
    activePara_s aPara = {
        .velocity = 1.0,
        .fluctuation = 0.5,
        .observeRadius = 1.0,
    };

    randomGen = mt19937(gPara.randomSeed);
    randomDist = uniform_real_distribution<float>(0,1);

    //initialize the data
    arrayPtr posX(new float[gPara.birdNum]);
    arrayPtr posY(new float[gPara.birdNum]);
    arrayPtr theta(new float[gPara.birdNum]);


    for(int i=0; i < gPara.birdNum; i++){
        //randomize the pos and theta
        auto randomFloat = randomDist(randomGen);
        posX[i] = randomFloat * gPara.fieldLength;

        randomFloat = randomDist(randomGen);
        posY[i] = randomFloat * gPara.fieldLength;

        randomFloat = randomDist(randomGen);
        theta[i] = randomFloat * M_PI * 2;
    }

    // timing the computing
    using namespace std::chrono;
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();

    //computing
    computeActiveMatter(gPara, aPara, posX, posY, theta);

    t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    cout << "Bird: " << gPara.birdNum << " Step: " << gPara.totalStep << " Compute Time: " << duration << "ms" << endl;

    delete[] posX;
    delete[] posY;
    delete[] theta;

    return 0;
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
int outputToFile(ofstream& outputFile, int birdNum, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta){
    //add current data to the file
    outputFile << "{" ;
    for(int bird=0; bird < birdNum; bird++){
        outputFile << posX[bird] << "," << posY[bird] << "," << theta[bird] << ";";
    }
    outputFile << "}" << endl;
    return 0;
}

/**
 * @brief computation of the activeMatter
 * 
 * @details compute position(posX, posY) and orientation(theta) all of the steps, 
 * output to file if enabled.
 * @param[in] gPara structure of general paramters 
 * @param[in] aPara structure of bird parameters
 * @param[in] posX reference of the pointer to the array of birds' position X
 * @param[in] posY reference of the pointer to the array of birds' position Y
 * @param[in] theta reference of the pointer to the array of birds' theta
*/
void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta){

    arrayPtr tempTheta(new float[gPara.birdNum]);
    ofstream outputFile;
    float observeRadiusSqr = pow(aPara.observeRadius,2);
    float inscribedSquareSideLengthHalf = aPara.observeRadius / sqrt(2);

    if(OUTPUT_TO_FILE){//save the parameter,first step to file
        outputFile = ofstream(gPara.outputPath, ios::trunc);
        if(outputFile.is_open()){
            outputFile << std::fixed << std::setprecision(3);
        }else{
            cout << "[!]Unable to open output file: " << gPara.outputPath << endl;
            exit(-1);
        }

        //write the params
        outputFile << "generalParameter{" 
        << "fieldLength=" << gPara.fieldLength 
        << ",totalStep=" << gPara.totalStep 
        << ",birdNum=" << gPara.birdNum 
        << ",randomSeed=" << gPara.randomSeed 
        << "}" << endl;
        //write the data of first step
        outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
    }

    for(int step=0; step < gPara.totalStep; step++){ //steps

        for(int bird=0; bird < gPara.birdNum; bird++){ //move
            //move
            posX[bird] += gPara.deltaTime * cos(theta[bird]);
            posY[bird] += gPara.deltaTime * sin(theta[bird]);

            //in the field
            posX[bird] = fmod(posX[bird]+gPara.fieldLength, gPara.fieldLength);
            posY[bird] = fmod(posY[bird]+gPara.fieldLength, gPara.fieldLength);

        }
        //adjust theta
        for(int bird=0; bird < gPara.birdNum; bird++){ //for each bird

            //float meanTheta = theta[bird];
            float sx = 0,sy = 0; 

            for(int oBird=0; oBird < gPara.birdNum; oBird++){ //observe other birds, self included

                auto xDiffAbs = abs(posX[bird]-posX[oBird]);
                auto yDiffAbs = abs(posY[bird]-posY[oBird]);
                
                if((xDiffAbs > aPara.observeRadius) || 
                    (yDiffAbs > aPara.observeRadius) 
                    || ((xDiffAbs > inscribedSquareSideLengthHalf) && (yDiffAbs > inscribedSquareSideLengthHalf))
                    )continue;//ignore birds outside the circumscribed square and 4 corners

                if((xDiffAbs < inscribedSquareSideLengthHalf) && 
                    (yDiffAbs < inscribedSquareSideLengthHalf)){ //birds inside the inscribed square
                    sx += cos(theta[oBird]);
                    sy += sin(theta[oBird]);
                }else{
                    auto distPow2 = pow(xDiffAbs, 2) + pow(yDiffAbs, 2);
                    if(distPow2 < observeRadiusSqr){ //observed
                        sx += cos(theta[oBird]);
                        sy += sin(theta[oBird]);
                    }
                }

                
            }
            tempTheta[bird] = atan2(sy, sx) + (randomDist(randomGen) - 0.5) * aPara.fluctuation; //new theta
        }
        //copy, could be dual-buffer
        //copy(tempTheta.get(), tempTheta.get()+gPara.birdNum, theta.get());
        //memcpy(theta.get(), tempTheta.get(), gPara.birdNum * sizeof(*theta.get())); //copy to theta

        //dual-buffer, swap ptr
        auto tempPtr = theta;
        theta = tempTheta;
        tempTheta = tempPtr;

        if(OUTPUT_TO_FILE)outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
        
    }

    if(OUTPUT_TO_FILE)outputFile.close();
    delete[] tempTheta;
}


