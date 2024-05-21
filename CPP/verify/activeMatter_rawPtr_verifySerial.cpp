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
mt19937 randomGen;
uniform_real_distribution<float> randomDist;

void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta);



int main(int argc, char* argv[]){

    
    generalPara_s gPara = {
            .fieldLength = 10.0,
            .deltaTime = 0.2,
            .totalStep = 500,
            .birdNum = 500,
            .randomSeed = static_cast<int>(time(nullptr)),
            .outputPath = "./output.plot",
        };

    if(argc != 5){
        printf("[!] 5 arguments required\n");
        exit(-1);
    }else{
        //total_step bird_num random_Seed verifyOutputName
        gPara.totalStep = atoi(argv[1]);
        gPara.birdNum = atoi(argv[2]);
        gPara.randomSeed = atoi(argv[3]);
        gPara.outputPath = argv[4];

        printf("[*] Start to generate for totalStep: %d, birdNum: %d, randomSeed: %d, Path: %s\n", 
                                    gPara.totalStep, gPara.birdNum, gPara.randomSeed, gPara.outputPath.c_str());
    }

    //load the params
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

        //para
        outputFile << "generalParameter{" 
        << "fieldLength=" << gPara.fieldLength 
        << ",totalStep=" << gPara.totalStep 
        << ",birdNum=" << gPara.birdNum 
        << ",randomSeed=" << gPara.randomSeed 
        << "}" << endl;
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


