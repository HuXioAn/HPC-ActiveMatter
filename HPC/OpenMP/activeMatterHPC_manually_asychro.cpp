/*!
 * \file activeMatterHPC_manually.cpp
 * \brief Parallel Active Matter simulation using OpenMP.
 * \author Yitong
 * \date 016/05/24
 */

#include <iostream>
#include <cmath>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <cstdio>
using namespace std;

constexpr int DEFAULT_BIRD_NUM = 500; 
constexpr bool OUTPUT_TO_FILE = true;
const int Max_Thread_Num = 256;

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

void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta,int threadNum);



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

    //computing
    //for(int i = 1; i <= Max_Thread_Num; i++)
    computeActiveMatter(gPara, aPara, posX, posY, theta, Max_Thread_Num);

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


void computeActiveMatter(generalPara_t gPara, activePara_t aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, int threadNum){
    arrayPtr bufTheta(new float[gPara.birdNum]);
    arrayPtr bufposX(new float[gPara.birdNum]);
    arrayPtr bufposY(new float[gPara.birdNum]);
    ofstream outputFile;
    float observeRadiusSqr = pow(aPara.observeRadius, 2);
    float inscribedSquareSideLengthHalf = aPara.observeRadius / sqrt(2);
    double start_time, end_time;
    //omp_set_num_threads(threadNum);
    start_time = omp_get_wtime();
    omp_set_nested(1);
    if (OUTPUT_TO_FILE) { // save the parameter, first step to file
        outputFile = ofstream(gPara.outputPath, ios::trunc);
        if (outputFile.is_open()) {
            outputFile << std::fixed << std::setprecision(3);
        } else {
            cout << "[!]Unable to open output file: " << gPara.outputPath << endl;
            exit(-1);
        }

        // para
        outputFile << "generalParameter{" << "fieldLength=" << gPara.fieldLength << ",totalStep=" << gPara.totalStep <<
            ",birdNum=" << gPara.birdNum << "}" << endl;
        // data
        // outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
    }
    threadNum--;
    // make the whole calculation part parallel
    #pragma omp parallel num_threads(2)
    {
        for (int step = 0; step < gPara.totalStep; step++) {
            #pragma omp sections
            {
                #pragma omp section
                {
                    #pragma omp parallel num_threads(threadNum)
                    {
                        int id = omp_get_thread_num();
                        // manually allocate threads
                        for (int bird = id; bird < gPara.birdNum; bird += threadNum) { // move
                            // move
                            bufposX[bird] = gPara.deltaTime * cos(theta[bird]) + posX[bird];
                            bufposY[bird] = gPara.deltaTime * sin(theta[bird]) + posY[bird];

                            // in the field
                            bufposX[bird] = fmod(bufposX[bird] + gPara.fieldLength, gPara.fieldLength);
                            bufposY[bird] = fmod(bufposY[bird] + gPara.fieldLength, gPara.fieldLength);

                        }
                        // use barrier to Synchronize
                        #pragma omp barrier
                        // adjust theta
                        for (int bird = id; bird < gPara.birdNum; bird += threadNum) { // for each bird
                            // float meanTheta = theta[bird];
                            float sx = 0, sy = 0;
                            for (int oBird = 0; oBird < gPara.birdNum; oBird++) { // observe other birds, self included

                                auto xDiffAbs = abs(bufposX[bird] - bufposX[oBird]);
                                auto yDiffAbs = abs(bufposY[bird] - bufposY[oBird]);

                                if ((xDiffAbs > aPara.observeRadius) ||
                                    (yDiffAbs > aPara.observeRadius)
                                    || ((xDiffAbs > inscribedSquareSideLengthHalf) && (yDiffAbs > inscribedSquareSideLengthHalf))
                                    ) continue; // ignore birds outside the circumscribed square and 4 corners
                                if ((xDiffAbs < inscribedSquareSideLengthHalf) &&
                                    (yDiffAbs < inscribedSquareSideLengthHalf)) { // birds inside the inscribed square
                                    sx += cos(theta[oBird]);
                                    sy += sin(theta[oBird]);
                                } else {
                                    auto distPow2 = pow(xDiffAbs, 2) + pow(yDiffAbs, 2);
                                    if (distPow2 < observeRadiusSqr) { // observed
                                        sx += cos(theta[oBird]);
                                        sy += sin(theta[oBird]);
                                    }
                                }
                            }
                            bufTheta[bird] = atan2(sy, sx) + (randomDist(randomGen) - 0.5) * aPara.fluctuation; // new theta
                        }
                    }
                }
                #pragma omp section
                {
                    if (OUTPUT_TO_FILE)
                        outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
                }
            }
            #pragma omp master
            {
                auto tempPtr = theta;
                theta = bufTheta;
                bufTheta = tempPtr;
                auto tempPtr2 = posX;
                posX = bufposX;
                bufposX = tempPtr2;
                auto tempPtr3 = posY;
                posY = bufposY;
                bufposY = tempPtr3;
            }

        }

    }
    end_time = omp_get_wtime();
    printf("Execution time of %d threads:%lf\n", threadNum + 1, end_time - start_time);
    if (OUTPUT_TO_FILE) outputFile.close();
    delete[] bufposX;
    delete[] bufposY;
    delete[] bufTheta;
}



