/*
cpp serial ActiveMatter simulation
Auther: Anton
Create Time: 02/04/24

*/

#include <iostream>
#include <cmath>
#include <random>

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

    //initialize the data
    srand(gPara.randomSeed);

    arrayPtr posX(new float(gPara.birdNum));
    arrayPtr posY(new float(gPara.birdNum));
    arrayPtr theta(new float(gPara.birdNum));

    mt19937 radomGen(gPara.randomSeed);
    uniform_real_distribution<float> radomDist(0,1);
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


}


