/*
cpp serial ActiveMatter simulation
Auther: Anton
Create Time: 02/04/24

*/

#include <iostream>
#include <cmath>
#include <random>
#include <cstring>

using namespace std;

constexpr int DEFAULT_BIRD_NUM = 500; 

typedef struct generalPara_s
{
    float fieldLength;
    float deltaTime;
    int totalStep;
    int birdNum;

    int randomSeed;

}generalPara_t;

typedef struct activePara_s
{
    float velocity;
    float fluctuation; //in radians
    float observeRadius;

}activePara_t;

using arrayPtr = shared_ptr<float>;

//0-1 float random
mt19937 radomGen;
uniform_real_distribution<float> radomDist;

void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr posX, arrayPtr posY, arrayPtr theta);

int main(int argc, char* argv[]){

    auto birdNum = DEFAULT_BIRD_NUM;
    if(argc > 0){
        birdNum = atoi(argv[1]);
        if(birdNum == 0)birdNum = DEFAULT_BIRD_NUM;
    }

    //load the params
    generalPara_s gPara = {
        .fieldLength = 10.0,
        .deltaTime = 0.2,
        .totalStep = 500,
        .birdNum = birdNum,
        .randomSeed = time(nullptr)
    };

    activePara_s aPara = {
        .fluctuation = 0.5,
        .observeRadius = 1.0,
        .velocity = 1.0
    };

    radomGen = mt19937(gPara.randomSeed);
    radomDist = uniform_real_distribution<float>(0,1);

    //initialize the data
    arrayPtr posX(new float(gPara.birdNum));
    arrayPtr posY(new float(gPara.birdNum));
    arrayPtr theta(new float(gPara.birdNum));


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


    return 0;
}


void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr posX, arrayPtr posY, arrayPtr theta){

    arrayPtr tempTheta(new float(gPara.birdNum));

    for(int step=0; step < gPara.totalStep; step++){ //steps

        for(int bird=0; bird < gPara.birdNum; bird++){ //move
            //move
            posX[bird] += gPara.deltaTime * cos(theta[bird]);
            posY[bird] += gPara.deltaTime * sin(theta[bird]);

            //in the field
            posX[bird] = fmod(posX[bird], gPara.fieldLength);
            posY[bird] = fmod(posY[bird], gPara.fieldLength);

        }
        //adjust theta
        for(int bird=0; bird < gPara.birdNum; bird++){ //for each bird

            //float meanTheta = theta[bird];
            float sx = 0,sy = 0; 

            for(int oBird=0; oBird < gPara.birdNum; oBird++){ //observe other birds, self included
                auto distPow2 = pow(posX[bird]-posX[oBird],2) + pow(posY[bird]-posY[oBird],2);
                if(distPow2 < pow(aPara.observeRadius,2)){ //observed
                    sx += cos(theta[oBird]);
                    sy += sin(theta[oBird]);
                }
            }
            tempTheta[bird] = atan2(sy, sx) + (radomDist(radomGen) - 0.5) * aPara.fluctuation; //new theta
        }
        //copy, could be dual-buffer
        copy(tempTheta.get(), tempTheta.get(), theta.get());
        //memcpy(theta.get(), tempTheta.get(), gPara.birdNum * sizeof(*theta.get())); //copy to theta
        
    }
}


