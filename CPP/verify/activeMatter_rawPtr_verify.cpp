/**
 * @file activeMatter_rawPtr_verify.cpp
 * @brief verification of an output file
 * @details Single-threaded cpp code for verification, compare data of very step and  sum up the total error
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

void computeActiveMatterOneStep(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, 
                                                                        arrayPtr& posXVerify, arrayPtr& posYVerify, arrayPtr& thetaVerify);

int compareElementInRange(generalPara_t gPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, 
                                                arrayPtr& posXVerify, arrayPtr& posYVerify, arrayPtr& thetaVerify, float range);

void getPara(ifstream& file, generalPara_t& gPara);

void getData(ifstream& file, generalPara_t gPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta);

int main(int argc, char* argv[]){

    // the default general parameters
    generalPara_s gPara = {
        .fieldLength = 10.0,
        .deltaTime = 0.2,
        .totalStep = 500,
        .birdNum = 500,
        .randomSeed = static_cast<int>(time(nullptr)),
        .outputPath = "./output.plot",
    };
    // the default bird parameters
    activePara_s aPara = {
        .velocity = 1.0,
        .fluctuation = 0.5,
        .observeRadius = 1.0,
    };
    
    // stream of the file to be verified
    ifstream file;

    if(argc != 2){
        printf("[!] 1 arguments required\n");
        exit(-1);
    }else{

        file = ifstream(argv[1]);
        if(!file){
            printf("[!] Can not open file: %s\n", argv[1]);
            exit(-1);
        }
        getPara(file, gPara);
        printf("[*] Start to verify with totalStep: %d, birdNum: %d, randomSeed: %d, Path: %s\n", 
                                    gPara.totalStep, gPara.birdNum, gPara.randomSeed, gPara.outputPath.c_str());
    }

    //data in one step from the file
    arrayPtr posX(new float[gPara.birdNum]);
    arrayPtr posY(new float[gPara.birdNum]);
    arrayPtr theta(new float[gPara.birdNum]);

    //initialize the verifacation data
    arrayPtr posXVerify(new float[gPara.birdNum]);
    arrayPtr posYVerify(new float[gPara.birdNum]);
    arrayPtr thetaVerify(new float[gPara.birdNum]);

    randomGen = mt19937(gPara.randomSeed);
    randomDist = uniform_real_distribution<float>(0,1);


    for(int i=0; i < gPara.birdNum; i++){
        //randomize the pos and theta
        auto randomFloat = randomDist(randomGen);
        posXVerify[i] = randomFloat * gPara.fieldLength;

        randomFloat = randomDist(randomGen);
        posYVerify[i] = randomFloat * gPara.fieldLength;

        randomFloat = randomDist(randomGen);
        thetaVerify[i] = randomFloat * M_PI * 2;
    }

    // get first data line
    getData(file, gPara, posX, posY, theta); 
    // verify the first step
    auto result = compareElementInRange(gPara, posX, posY, theta, posXVerify, posYVerify, thetaVerify, 0.5 * aPara.fluctuation);
    if(result != 0){
        printf("[!] Verification failed in step %d\n", 0);
        exit(-1);
    }
    int totalErrorCount = 0;
    for(int i = 0; i < gPara.totalStep; i++){
        // compare every step
        // compute verification data with previous verified line from file
        computeActiveMatterOneStep(gPara, aPara, posX, posY, theta, posXVerify, posYVerify, thetaVerify);

        getData(file, gPara, posX, posY, theta);// new step from file
        result = compareElementInRange(gPara, posX, posY, theta, posXVerify, posYVerify, thetaVerify, 0.6 * aPara.fluctuation);
        if(result > 10){
            
            printf("[!] Verification failed in step %d with %d errors\n", i+1, result);
            exit(-1);
            
        }

        totalErrorCount += result;
    }

    printf("[*] Verification ended with %d errors in total\n", totalErrorCount);
    
    file.close();
    delete[] posX;
    delete[] posY;
    delete[] theta;

    delete[] posXVerify;
    delete[] posYVerify;
    delete[] thetaVerify;

    return 0;
}


/**
 * @brief computation of one step from the given state
 * 
 * @details compute one step from the given state(from the output file), however the thetaVerify contains no fluctuation
 * @param[in] gPara structure of general paramters 
 * @param[in] aPara structure of bird parameters
 * 
 * @param[in] posX reference of the pointer to the array of birds' last position X
 * @param[in] posY reference of the pointer to the array of birds' last position Y
 * @param[in] theta reference of the pointer to the array of birds' last theta
 * 
 * @param[out] posXVerify reference of the pointer to the array of computed birds' position X
 * @param[out] posYVerify reference of the pointer to the array of computed birds' position Y
 * @param[out] thetaVerify reference of the pointer to the array of computed birds' theta
*/
void computeActiveMatterOneStep(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, 
                                                                arrayPtr& posXVerify, arrayPtr& posYVerify, arrayPtr& thetaVerify){


    float observeRadiusSqr = pow(aPara.observeRadius,2);
    float inscribedSquareSideLengthHalf = aPara.observeRadius / sqrt(2);

    for(int bird=0; bird < gPara.birdNum; bird++){ //move
        //move
        posXVerify[bird] = posX[bird] + gPara.deltaTime * cos(theta[bird]);
        posYVerify[bird] = posY[bird] + gPara.deltaTime * sin(theta[bird]);

        //in the field
        posXVerify[bird] = fmod(posXVerify[bird]+gPara.fieldLength, gPara.fieldLength);
        posYVerify[bird] = fmod(posYVerify[bird]+gPara.fieldLength, gPara.fieldLength);

    }
    //adjust theta
    for(int bird=0; bird < gPara.birdNum; bird++){ //for each bird

        //float meanTheta = theta[bird];
        float sx = 0,sy = 0; 

        for(int oBird=0; oBird < gPara.birdNum; oBird++){ //observe other birds, self included

            auto xDiffAbs = abs(posXVerify[bird]-posXVerify[oBird]);
            auto yDiffAbs = abs(posYVerify[bird]-posYVerify[oBird]);
            
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
        thetaVerify[bird] = atan2(sy, sx); //new theta
    }


}

/**
 * @brief get parameters from the output file
 * 
 * @param[in] file output file stream
 * @param[out] gPara reference of the general parameter structure
*/
void getPara(ifstream& file, generalPara_t& gPara){

    string paraLine; 
    getline(file, paraLine);

    //get the params 
    size_t start = paraLine.find("fieldLength=") + 12;
    size_t end = paraLine.find(',', start);
    gPara.fieldLength = stoi(paraLine.substr(start, end - start));

    start = paraLine.find("totalStep=", end) + 10;
    end = paraLine.find(',', start);
    gPara.totalStep = stoi(paraLine.substr(start, end - start));

    start = paraLine.find("birdNum=", end) + 8;
    end = paraLine.find(',', start);
    gPara.birdNum = stoi(paraLine.substr(start, end - start));

    start = paraLine.find("randomSeed=", end) + 11;
    end = paraLine.find('}', start);
    gPara.randomSeed = stoul(paraLine.substr(start, end - start));
    

}

/**
 * @brief get one step data from the output file
 * 
 * @param[in] file output file stream
 * @param[in] gPara general parameter structure
 * 
 * @param[out] posX reference of the pointer to the array of birds' position X
 * @param[out] posY reference of the pointer to the array of birds' position Y
 * @param[out] theta reference of the pointer to the array of birds' theta
*/
void getData(ifstream& file, generalPara_t gPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta){
    string dataLine; 
    
    if (getline(file, dataLine)) {
        stringstream ss(dataLine.substr(1, dataLine.length() - 2));
        string segment;
        
        int n = 0;
        while (getline(ss, segment, ';')) {
            stringstream segmentStream(segment);
            char comma;
            float x, y, t;
            segmentStream >> x >> comma >> y >> comma >> t;

            posX[n] = x;
            posY[n] = y;
            theta[n] = t;
            n++;
        }
    }

}

/**
 * @brief compare one step data from target file and the computed ones
 * 
 * @details compare the position and the theta between the target file and the computed data against certain range
 * @param[in] gPara structure of general paramters 
 * 
 * @param[in] posX reference of the pointer to the array of birds' position X
 * @param[in] posY reference of the pointer to the array of birds' position Y
 * @param[in] theta reference of the pointer to the array of birds' theta
 * 
 * @param[in] posXVerify reference of the pointer to the array of computed birds' position X
 * @param[in] posYVerify reference of the pointer to the array of computed birds' position Y
 * @param[in] thetaVerify reference of the pointer to the array of computed birds' theta
*/
int compareElementInRange(generalPara_t gPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, 
                                                arrayPtr& posXVerify, arrayPtr& posYVerify, arrayPtr& thetaVerify, float range){

    int error = 0;
    for(int i = 0; i < gPara.birdNum; i++){
        //in case of field tranversing
        if (((abs(posX[i] - posXVerify[i]) > range) && !((gPara.fieldLength - abs(posX[i] - posXVerify[i])) < range)) ||
            ((abs(posY[i] - posYVerify[i]) > range) && !((gPara.fieldLength - abs(posY[i] - posYVerify[i])) < range)) ||
            abs(theta[i] - thetaVerify[i]) > range) {
            error++;
        }
    }

    return error;
}

