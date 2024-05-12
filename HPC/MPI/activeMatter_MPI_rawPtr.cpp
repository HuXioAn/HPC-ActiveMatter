#include <iostream>
#include <cmath>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>
#include <mpi.h>

using namespace std;

constexpr int DEFAULT_BIRD_NUM = 500;
constexpr bool OUTPUT_TO_FILE = true;

struct generalPara_s
{
    float fieldLength;
    float deltaTime;
    int totalStep;
    int birdNum;

    int randomSeed;
    string outputPath;
};

struct activePara_s
{
    float velocity;
    float fluctuation; // in radians
    float observeRadius;
};

using arrayPtr = float*;

// 0-1 float random
mt19937 randomGen;
uniform_real_distribution<float> randomDist;

void computeActiveMatter(generalPara_s gPara, activePara_s aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, int rank, int size);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int birdNum = DEFAULT_BIRD_NUM;
    if (argc > 1)
    {
        birdNum = atoi(argv[1]);
        if (birdNum <= 0)
            birdNum = DEFAULT_BIRD_NUM;
    }
    

    generalPara_s gPara = {
        .fieldLength = 10.0,
        .deltaTime = 0.2,
        .totalStep = 500,
        .birdNum = (birdNum / size) * size, // Ensure total birds is divisible by size
        .randomSeed = static_cast<int>(time(nullptr)),
        .outputPath = "./output.plot",
    };

    activePara_s aPara = {
        .velocity = 1.0,
        .fluctuation = 0.5,
        .observeRadius = 1.0,
    };

    randomGen.seed(gPara.randomSeed);
    randomDist = uniform_real_distribution<float>(0, 1);


    arrayPtr posX(new float[gPara.birdNum]);
    arrayPtr posY(new float[gPara.birdNum]);
    arrayPtr theta(new float[gPara.birdNum]);

    double startTime = 0;
    if (rank == 0)
    {
        for (int i = 0; i < gPara.birdNum; i++)
        {
            posX[i] = randomDist(randomGen) * gPara.fieldLength;
            posY[i] = randomDist(randomGen) * gPara.fieldLength;
            theta[i] = randomDist(randomGen) * 2 * M_PI;
        }
        startTime = MPI_Wtime();
    }

    MPI_Bcast(posX, gPara.birdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(posY, gPara.birdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(theta, gPara.birdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);


    computeActiveMatter(gPara, aPara, posX, posY, theta, rank, size);

    if(rank == 0){
        printf("With %d Process, Execution time: %lfs. \n", size, MPI_Wtime()-startTime);
    }
    MPI_Finalize();
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



void computeActiveMatter(generalPara_s gPara, activePara_s aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, int rank, int size)
{
    int localBirdNum = gPara.birdNum / size;
    arrayPtr tempTheta(new float[localBirdNum]);
    int localBirdOffset = localBirdNum * rank;

    float observeRadiusSqr = pow(aPara.observeRadius,2);
    float inscribedSquareSideLengthHalf = aPara.observeRadius / sqrt(2);

    ofstream outputFile;


    if(OUTPUT_TO_FILE && rank == 0){//save the parameter,first step to file
        outputFile = ofstream(gPara.outputPath, ios::trunc);
        if(outputFile.is_open()){
            outputFile << std::fixed << std::setprecision(3);
        }else{
            cout << "[!]Unable to open output file: " << gPara.outputPath << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        //para
        outputFile << "generalParameter{" << "fieldLength=" << gPara.fieldLength << ",totalStep=" << gPara.totalStep << 
            ",birdNum=" << gPara.birdNum << "}" << endl;
        //data
        outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
    }


    for (int step = 0; step < gPara.totalStep; step++)
    {

        for (int bird = localBirdOffset; bird < (localBirdNum + localBirdOffset); bird++)
        {
            // Move
            posX[bird] += gPara.deltaTime * cos(theta[bird]);
            posY[bird] += gPara.deltaTime * sin(theta[bird]);

            // in the field
            posX[bird] = fmod(posX[bird] + gPara.fieldLength, gPara.fieldLength);
            posY[bird] = fmod(posY[bird] + gPara.fieldLength, gPara.fieldLength);
        }

        MPI_Allgather(posX + localBirdOffset, localBirdNum, MPI_FLOAT, posX, localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(posY + localBirdOffset, localBirdNum, MPI_FLOAT, posY, localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);

        // Adjust theta
        for (int bird = localBirdOffset; bird < (localBirdNum + localBirdOffset); bird++)
        {
            float sx = 0, sy = 0;
            for (int oBird = 0; oBird < gPara.birdNum; oBird++)
            {
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
            tempTheta[bird - localBirdOffset] = atan2(sy, sx) + (randomDist(randomGen) - 0.5) * aPara.fluctuation;
        }

        MPI_Allgather(tempTheta, localBirdNum, MPI_FLOAT, theta, localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);

        if (OUTPUT_TO_FILE && rank == 0)
        {
            outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
        }
    }

    if (OUTPUT_TO_FILE && rank == 0)
        outputFile.close();
}