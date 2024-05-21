/**
 * @file activeMatter_MPI_rawPtr.cpp
 * @brief MPI parallel code for activeMatter
 * @details cpp code for activeMatter, all optimization applied
 * @author Andong Hu
 * @author Guoqing Liang
 * @date 2024-5-21
 */

#include <iostream>
#include <cmath>
#include <random>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>
#include <mpi.h>

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

void computeActiveMatter(generalPara_s gPara, activePara_s aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, int rank, int size);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size; //mpi process id and total number
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int birdNum = DEFAULT_BIRD_NUM;
    if (argc > 1)
    {
        birdNum = atoi(argv[1]);
        if (birdNum <= 0)
            birdNum = DEFAULT_BIRD_NUM;
    }
    
    // the default general parameters
    generalPara_s gPara = {
        .fieldLength = 10.0,
        .deltaTime = 0.2,
        .totalStep = 500,
        .birdNum = (birdNum / size) * size, // Ensure total birds is divisible by size
        .randomSeed = static_cast<int>(time(nullptr)),
        .outputPath = "./output.plot",
    };

    // the default bird parameters
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
    if (rank == 0) // proc 0 initialize the data
    {
        for (int i = 0; i < gPara.birdNum; i++)
        {
            posX[i] = randomDist(randomGen) * gPara.fieldLength;
            posY[i] = randomDist(randomGen) * gPara.fieldLength;
            theta[i] = randomDist(randomGen) * 2 * M_PI;
        }
        startTime = MPI_Wtime();
    }
    // send initialized data to all procs
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
void computeActiveMatter(generalPara_s gPara, activePara_s aPara, arrayPtr& posX, arrayPtr& posY, arrayPtr& theta, int rank, int size)
{
    // bird number for every process
    int localBirdNum = gPara.birdNum / size;
    arrayPtr tempTheta(new float[localBirdNum]);
    // the offset in the arrays for this certain proc
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
        outputFile << "generalParameter{" 
        << "fieldLength=" << gPara.fieldLength 
        << ",totalStep=" << gPara.totalStep 
        << ",birdNum=" << gPara.birdNum 
        << ",randomSeed=" << gPara.randomSeed 
        << "}" << endl;
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
        // sync all the position data
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
        // sync all the theta
        MPI_Allgather(tempTheta, localBirdNum, MPI_FLOAT, theta, localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);

        if (OUTPUT_TO_FILE && rank == 0)
        {
            outputToFile(outputFile, gPara.birdNum, posX, posY, theta);
        }
    }

    if (OUTPUT_TO_FILE && rank == 0)
        outputFile.close();
}