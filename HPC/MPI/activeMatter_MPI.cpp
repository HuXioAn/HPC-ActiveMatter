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

// 0-1 float random
mt19937 randomGen;
uniform_real_distribution<float> randomDist;

void computeActiveMatter(generalPara_s gPara, activePara_s aPara, vector<float> &posX, vector<float> &posY, vector<float> &theta, int rank, int size);

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
    birdNum = (birdNum / size) * size; // Ensure total birds is divisible by size

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

    randomGen.seed(gPara.randomSeed);
    randomDist = uniform_real_distribution<float>(0, 1);

    int localBirdNum = gPara.birdNum / size;
    vector<float> posX(localBirdNum), posY(localBirdNum), theta(localBirdNum);

    // Generate initial conditions on root and distribute
    vector<float> allPosX(gPara.birdNum), allPosY(gPara.birdNum), allTheta(gPara.birdNum);
    double startTime = 0;
    if (rank == 0)
    {
        for (int i = 0; i < gPara.birdNum; i++)
        {
            allPosX[i] = randomDist(randomGen) * gPara.fieldLength;
            allPosY[i] = randomDist(randomGen) * gPara.fieldLength;
            allTheta[i] = randomDist(randomGen) * 2 * M_PI;
        }
        startTime = MPI_Wtime();
    }
    MPI_Scatter(allPosX.data(), localBirdNum, MPI_FLOAT, posX.data(), localBirdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(allPosY.data(), localBirdNum, MPI_FLOAT, posY.data(), localBirdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(allTheta.data(), localBirdNum, MPI_FLOAT, theta.data(), localBirdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
    computeActiveMatter(gPara, aPara, posX, posY, theta, rank, size);

    if(rank == 0){
        printf("With %d Process, Execution time: %lf", size, MPI_Wtime()-startTime);
    }
    MPI_Finalize();
    return 0;
}

int outputToFile(ofstream &outputFile, vector<float> &posX, vector<float> &posY, vector<float> &theta)
{
    // Add current data to the file
    outputFile << "{";
    for (size_t bird = 0; bird < posX.size(); bird++)
    {
        outputFile << posX[bird] << "," << posY[bird] << "," << theta[bird] << ";";
    }
    outputFile << "}" << endl;
    return 0;
}

void computeActiveMatter(generalPara_s gPara, activePara_s aPara, vector<float> &posX, vector<float> &posY, vector<float> &theta, int rank, int size)
{
    int localBirdNum = gPara.birdNum / size;
    vector<float> tempTheta(localBirdNum);
    ofstream outputFile;
    vector<float> allPosX(gPara.birdNum), allPosY(gPara.birdNum), allTheta(gPara.birdNum);

    if (rank == 0)
    {
        outputFile.open(gPara.outputPath, ios::trunc);
        if (outputFile.is_open())
        {
            outputFile << std::fixed << std::setprecision(3);
            outputFile << "generalParameter{"
                       << "fieldLength=" << gPara.fieldLength << ",totalStep=" << gPara.totalStep
                       << ",birdNum=" << gPara.birdNum << "}" << endl;
        }
        else
        {
            cout << "[!] Unable to open output file: " << gPara.outputPath << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    // Gather positions to all processes
    MPI_Allgather(posX.data(), localBirdNum, MPI_FLOAT, allPosX.data(), localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(posY.data(), localBirdNum, MPI_FLOAT, allPosY.data(), localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(theta.data(), localBirdNum, MPI_FLOAT, allTheta.data(), localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);

    for (int step = 0; step < gPara.totalStep; step++)
    {

        for (int bird = 0; bird < localBirdNum; bird++)
        {
            // Move
            posX[bird] += gPara.deltaTime * cos(theta[bird]);
            posY[bird] += gPara.deltaTime * sin(theta[bird]);

            // in the field
            posX[bird] = fmod(posX[bird] + gPara.fieldLength, gPara.fieldLength);
            posY[bird] = fmod(posY[bird] + gPara.fieldLength, gPara.fieldLength);
        }

        MPI_Allgather(posX.data(), localBirdNum, MPI_FLOAT, allPosX.data(), localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(posY.data(), localBirdNum, MPI_FLOAT, allPosY.data(), localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgather(theta.data(), localBirdNum, MPI_FLOAT, allTheta.data(), localBirdNum, MPI_FLOAT, MPI_COMM_WORLD);

        // Adjust theta
        for (int bird = 0; bird < localBirdNum; bird++)
        {
            float sx = 0, sy = 0;
            for (int oBird = 0; oBird < gPara.birdNum; oBird++)
            {
                if ((abs(posX[bird] - allPosX[oBird]) > aPara.observeRadius) ||
                    (abs(posY[bird] - allPosY[oBird]) > aPara.observeRadius))
                    continue;
                // auto distPow2 = pow(posX[bird] - allPosX[oBird], 2) + pow(posY[bird] - allPosY[oBird], 2);
                if (pow(posX[bird] - allPosX[oBird], 2) + pow(posY[bird] - allPosY[oBird], 2) < pow(aPara.observeRadius, 2))
                { // observed
                    sx += cos(allTheta[oBird]);
                    sy += sin(allTheta[oBird]);
                }
            }
            // tempTheta[bird] = atan2(sy, sx) + (randomDist(randomGen) - 0.5) * aPara.fluctuation;
            tempTheta[bird] = atan2(sy, sx) + (randomDist(randomGen) - 0.5) * aPara.fluctuation;
        }
        theta = tempTheta;

        vector<float> allPosX(gPara.birdNum), allPosY(gPara.birdNum), allTheta(gPara.birdNum);
        MPI_Gather(posX.data(), localBirdNum, MPI_FLOAT, allPosX.data(), localBirdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(posY.data(), localBirdNum, MPI_FLOAT, allPosY.data(), localBirdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gather(theta.data(), localBirdNum, MPI_FLOAT, allTheta.data(), localBirdNum, MPI_FLOAT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            outputToFile(outputFile, allPosX, allPosY, allTheta);
        }
    }

    if (rank == 0)
        outputFile.close();
}