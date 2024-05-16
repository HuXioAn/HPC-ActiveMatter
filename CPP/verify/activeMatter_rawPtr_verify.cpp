/*
cpp serial ActiveMatter simulation
Auther: Anton
Create Time: 02/04/24

*/

#include <iostream>
#include <cmath>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace std;

constexpr bool OUTPUT_TO_FILE = false;

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

using arrayPtr = float*;

//0-1 float random
mt19937 randomGen;
uniform_real_distribution<float> randomDist;

void computeActiveMatterOneStep(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, 
                                                                        arrayPtr& posXVerify, arrayPtr& posYVerify, arrayPtr& thetaVerify);

int compareElementInRange(generalPara_t gPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, 
                                                arrayPtr& posXVerify, arrayPtr& posYVerify, arrayPtr& thetaVerify, float range);

void getPara(ifstream& file, generalPara_t& gPara);

void getData(ifstream& file, generalPara_t gPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta);

int main(int argc, char* argv[]){

    
    generalPara_s gPara = {
        .fieldLength = 10.0,
        .deltaTime = 0.2,
        .totalStep = 500,
        .birdNum = 500,
        .randomSeed = static_cast<int>(time(nullptr)),
        .outputPath = "./output.plot",
    };

    activePara_s aPara = {
        .velocity = 1.0,
        .fluctuation = 0.5,
        .observeRadius = 1.0,
    };
    
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

    getData(file, gPara, posX, posY, theta);//get first data line
    auto result = compareElementInRange(gPara, posX, posY, theta, posXVerify, posYVerify, thetaVerify, 0.5 * aPara.fluctuation);
    if(result != 0){
        printf("[!] Verification failed in step %d\n", 0);
        exit(-1);
    }

    for(int i = 0; i < gPara.totalStep; i++){
        //compare every step

        computeActiveMatterOneStep(gPara, aPara, posX, posY, theta, posXVerify, posYVerify, thetaVerify);//compute verification data with previous verified line from file

        getData(file, gPara, posX, posY, theta);//new step from file
        result = compareElementInRange(gPara, posX, posY, theta, posXVerify, posYVerify, thetaVerify, 1 * aPara.fluctuation);
        if(result != 0){
            printf("[!] Verification failed in step %d\n", i+1);
            exit(-1);
        }

    }
    



    delete[] posX;
    delete[] posY;
    delete[] theta;

    delete[] posXVerify;
    delete[] posYVerify;
    delete[] thetaVerify;

    return 0;
}



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


int compareElementInRange(generalPara_t gPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, 
                                                arrayPtr& posXVerify, arrayPtr& posYVerify, arrayPtr& thetaVerify, float range){


    for(int i = 0; i < gPara.birdNum; i++){
        if (((abs(posX[i] - posXVerify[i]) > range) && !(abs(posX[i] + posXVerify[i] - gPara.fieldLength) < range)) ||
            ((abs(posY[i] - posYVerify[i]) > range) && !(abs(posY[i] + posYVerify[i] - gPara.fieldLength) < range)) ||
            abs(theta[i] - thetaVerify[i]) > range) {
            return 1;
        }
    }

    return 0;
}

