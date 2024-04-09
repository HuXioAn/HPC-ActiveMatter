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

using arrayPtr = float*;

//0-1 float random
mt19937 radomGen;
uniform_real_distribution<float> radomDist;

void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta);



int main(int argc, char* argv[]){

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

    radomGen = mt19937(gPara.randomSeed);
    radomDist = uniform_real_distribution<float>(0,1);

    //initialize the data
    arrayPtr posX(new float[gPara.birdNum]);
    arrayPtr posY(new float[gPara.birdNum]);
    arrayPtr theta(new float[gPara.birdNum]);


    for(int i=0; i < gPara.birdNum; i++){
        //randomize the pos and theta
        auto randomFloat = radomDist(radomGen);
        posX[i] = randomFloat * gPara.fieldLength;

        randomFloat = radomDist(radomGen);
        posY[i] = randomFloat * gPara.fieldLength;

        randomFloat = radomDist(radomGen);
        theta[i] = randomFloat * M_PI * 2;
    }

    //computing
    
    computeActiveMatter(gPara, aPara, posX, posY, theta);

    delete[] posX;
    delete[] posY;
    delete[] theta;

    return 0;
}

int outputToFile(ofstream& outputFile, int birdNum, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta){
    //add current data to the file
    outputFile << "{" ;
    for(int bird=0; bird < birdNum; bird++){
        outputFile << posX[bird] << "," << posY[bird] << "," << theta[bird] << ";";
    }
    outputFile << "}" << endl;
    return 0;
}


void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta){

    arrayPtr tempTheta(new float[gPara.birdNum]);
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
                if((abs(posX[bird]-posX[oBird]) > aPara.observeRadius) || 
                    (abs(posY[bird]-posY[oBird]) > aPara.observeRadius))continue;
                auto distPow2 = pow(posX[bird]-posX[oBird],2) + pow(posY[bird]-posY[oBird],2);
                if(distPow2 < pow(aPara.observeRadius,2)){ //observed
                    sx += cos(theta[oBird]);
                    sy += sin(theta[oBird]);
                }
            }
            tempTheta[bird] = atan2(sy, sx) + (radomDist(radomGen) - 0.5) * aPara.fluctuation; //new theta
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


