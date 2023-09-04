#include <iostream>
#include <math.h>
#include <vector>
#include "random_normal.h"
#include <mpi.h>

using namespace std;

#define DEBUG_BREAK(x)                  \
    {                                   \
        cout << "DEBUG #" << x << endl; \
    }
/*
Construction of a Random Walk

We need to construct the value of (X(t1), X(t2), ... , X(tn)) for the fixed set of the points 0 < t1 < t2 < ... < tn
We use the cholesky decomposition for the faster computation of the multi-variate normals

Brownian Bridge Construction (for faster construction -> Binary takes log2(n))


Geometric Brownian Motion with Ito's lemma

Assumptions:

The expiration time is T and we consider N different time segments each of length T/N
for a strike price of K devise a framework to price asian options
MPI Doesn't support global memory by default
*/
/*
One way to make things fast is to use a parallelistion library to calculate a random vector

*/
int indexVector[1000];
vector<float> getNormalVector(int size_of_normal_vec)
{
    // get a vector of size n
    // get the number of preocessors
    int pid, np, elements_per_process, n_elements_recieved, start_index;

    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // initialise a shared array
    int data_size = size_of_normal_vec; // Size of the shared array
    // float *shared_array = (float *)malloc(sizeof(float) * data_size);

    // Allocate shared memory using MPI_Win_allocate_shared
    float *shared_array = (float *)malloc(sizeof(float) * data_size);
    MPI_Win win;
    MPI_Win_allocate_shared(sizeof(float) * data_size, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &shared_array, &win);

    if (pid == 0)
    {
        // this is the master process
        int index, iter;
        elements_per_process = data_size / np;

        // initialise the array
        // if more than one processes are running
        if (np > 1)
        {
            for (iter = 1; iter < np - 1; iter++)
            {
                index = iter * elements_per_process; // this is the starting index of the subprocess
                indexVector[index] = index;
                MPI_Send(&elements_per_process, 1, MPI_INT, iter, 0, MPI_COMM_WORLD);
                MPI_Send(&indexVector[index], 1, MPI_INT, iter, 0, MPI_COMM_WORLD);
            }

            // the last process does it for the remaining elements
            index = iter * elements_per_process;
            int elements_left = data_size - index;
            indexVector[index] = index;
            MPI_Send(&elements_left, 1, MPI_INT, iter, 0, MPI_COMM_WORLD);
            MPI_Send(&indexVector[index], 1, MPI_INT, iter, 0, MPI_COMM_WORLD);
        }

        // master process creates its own random variables
        for (iter = 0; iter < elements_per_process; iter++)
        {
            CustomRandGenerator custRand;
            custRand.init(0);
            BoxMuller normGen(custRand);
            float value_to_put = normGen.getNormal();
            // puts the data from the corresponsing pid to the target shared memory at the corresponding location
            MPI_Put(&value_to_put, 1, MPI_FLOAT, 0, iter, 1, MPI_FLOAT, win);
        }
    }
    else
    {
        MPI_Recv(&n_elements_recieved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&start_index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        for (int iter = 0; iter < n_elements_recieved; iter++)
        {
            CustomRandGenerator custRand;
            custRand.init(0);
            BoxMuller normGen(custRand);
            // shared_array[start_index + iter] = normGen.getNormal();
            float value_to_put = normGen.getNormal();
            MPI_Put(&value_to_put, 1, MPI_FLOAT, 0, start_index + iter, 1, MPI_FLOAT, win);
        }
    }
    // MPI_Accumulate( const void* origin_addr , MPI_Count origin_count , MPI_Datatype origin_datatype , int target_rank , MPI_Aint target_disp , MPI_Count target_count , MPI_Datatype target_datatype , MPI_Op op , MPI_Win win);
    MPI_Barrier(MPI_COMM_WORLD);
    vector<float> retVec;
    for (int i = 0; i < data_size; i++)
    {
        retVec.push_back(shared_array[i]);
    }
    MPI_Win_free(&win);
    return retVec;
    // if(pid == 0) serialProcessDriver(shared_array, size_of_normal_vec);
}

float calculate_payoff_discounted(vector<float> stock_price, int N, float T, float r_det, float strike)
{
    assert(stock_price.size() == N); // need all the values of the stock prices

    // find the average of all the prices
    float avg_of_stock_price = 0;

    for (int _iter = 0; _iter < N; _iter++)
    {
        // update the average value
        avg_of_stock_price = (avg_of_stock_price + (stock_price[_iter] - avg_of_stock_price) / (_iter + 1));
    }

    float discounted_payoff_path = exp(-r_det * T) * max(avg_of_stock_price - strike, 0.0f);

    return discounted_payoff_path;
}

vector<float> MC_generate_path_MPI(float T, int N, float vol_det, float mu, float S0)
{
    // mu is the drift of the GMB while the drift of the log(S) is mu - (S^2) / 2
    // use a recursive function to get the path of the stock price
    //  the stock follows a geometric brownian motion
    // S(ti + 1) = S(ti)exp((mu - S^2 / 2)(dt) + S*sqrt(dt)*Z) where dt = t(i+1) - t(i)

    // use the box muller method to get normally distributed
    // this has been optimized for MPI Implemenatation
    vector<float> price_path(N);
    price_path[0] = S0;
    float dt = T / N; // dt when all the time periods are the same for easier computational complexity
    float dt_sqrt = sqrt(dt);
    float vol_det_pow_2 = pow(vol_det, 2);
    vector<float> normal_vector_curr = getNormalVector(N - 1); // this is a parallel step that will take place
    for (int i = 1; i < N; i++)
    {
        float Z_normal = normal_vector_curr[i - 1];
        price_path[i] = price_path[i - 1] * exp((mu - vol_det_pow_2 / 2) * dt + vol_det * dt_sqrt * Z_normal);
    }

    return price_path;
}
vector<float> MC_generate_path(float T, int N, float vol_det, float mu, float S0)
{
    CustomRandGenerator randGen;
    randGen.init(0);
    BoxMuller bx(randGen);

    vector<float> price_path(N);
    price_path[0] = S0;
    float dt = T / N; // dt when all the time periods are the same for easier computational complexity
    float dt_sqrt = sqrt(dt);
    float vol_det_pow_2 = pow(vol_det, 2);
    for (int i = 1; i < N; i++)
    {
        float Z_normal = bx.getNormal();
        price_path[i] = price_path[i - 1] * exp((mu - vol_det_pow_2 / 2) * dt + vol_det * dt_sqrt * Z_normal);
    }

    return price_path;
}

float MC_simulation_driver(float T, int N, float S0, float vol_det, float mu, float r_det, float strike)
{
    // reduce computational time for the number of simulations by multi-threading (high complexity of heirarchy)
    // simulating this is much better

    const int NUM_OF_SIMULATIONS = 10000;

    float expected_price = 0;

    for (int _iter = 0; _iter < NUM_OF_SIMULATIONS; _iter++)
    {
        vector<float> path = MC_generate_path_MPI(T, N, vol_det, mu, S0);

        float options_price_sim = calculate_payoff_discounted(path, N, T, r_det, strike);

        expected_price = (expected_price + (options_price_sim - expected_price) / (_iter + 1));
    }

    return expected_price;
}
vector<float> MC_simulation_driver_MPI(float T, int N, float S0, float vol_det, float mu, float r_det, float strike, int number_of_simulations)
{
    // get a vector of size n
    // get the number of preocessors
    int pid, np, elements_per_process, n_elements_recieved, start_index;

    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    // initialise a shared array
    int data_size = number_of_simulations; // Size of the shared array
    // float *shared_array = (float *)malloc(sizeof(float) * data_size);

    // Allocate shared memory using MPI_Win_allocate_shared
    float *shared_array = (float *)malloc(sizeof(float) * (data_size + 1));
    MPI_Win win;
    MPI_Win_allocate_shared(sizeof(float) * (data_size + 1), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &shared_array, &win);
    //last index of the shared memory contains the average
    if (pid == 0)
    {
        // this is the master process
        int index, iter;
        elements_per_process = data_size / np;

        // initialise the array
        // if more than one processes are running
        if (np > 1)
        {
            for (iter = 1; iter < np - 1; iter++)
            {
                index = iter * elements_per_process; // this is the starting index of the subprocess
                indexVector[index] = index;
                MPI_Send(&elements_per_process, 1, MPI_INT, iter, 0, MPI_COMM_WORLD);
                MPI_Send(&indexVector[index], 1, MPI_INT, iter, 0, MPI_COMM_WORLD);
            }

            // the last process does it for the remaining elements
            index = iter * elements_per_process;
            int elements_left = data_size - index;
            indexVector[index] = index;
            MPI_Send(&elements_left, 1, MPI_INT, iter, 0, MPI_COMM_WORLD);
            MPI_Send(&indexVector[index], 1, MPI_INT, iter, 0, MPI_COMM_WORLD);
        }

        // master process creates its own random variables
        for (iter = 0; iter < elements_per_process; iter++)
        {
            vector<float> path = MC_generate_path(T, N, vol_det, mu, S0);
            float options_price_sim = calculate_payoff_discounted(path, N, T, r_det, strike);
            // puts the data from the corresponsing pid to the target shared memory at the corresponding location
            MPI_Put(&options_price_sim, 1, MPI_FLOAT, 0, iter, 1, MPI_FLOAT, win);
            MPI_Accumulate( &options_price_sim , 1 , MPI_FLOAT , 0 , data_size , 1 , MPI_FLOAT , MPI_SUM , win);
        }
    }
    else
    {
        MPI_Recv(&n_elements_recieved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&start_index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        for (int iter = 0; iter < n_elements_recieved; iter++)
        {
            vector<float> path = MC_generate_path(T, N, vol_det, mu, S0);
            float options_price_sim = calculate_payoff_discounted(path, N, T, r_det, strike);
            MPI_Put(&options_price_sim, 1, MPI_FLOAT, 0, start_index + iter, 1, MPI_FLOAT, win);
            MPI_Accumulate( &options_price_sim , 1 , MPI_FLOAT , 0 , data_size , 1 , MPI_FLOAT , MPI_SUM , win);
        }
    }
    // MPI_Accumulate( const void* origin_addr , MPI_Count origin_count , MPI_Datatype origin_datatype , int target_rank , MPI_Aint target_disp , MPI_Count target_count , MPI_Datatype target_datatype , MPI_Op op , MPI_Win win);
    MPI_Barrier(MPI_COMM_WORLD);
    vector<float> retVec(data_size + 1);
    for (int i = 0; i <= data_size; i++)
    {
        retVec[i] = shared_array[i];
    }
    MPI_Win_free(&win);
    return retVec;
    // if(pid == 0) serialProcessDriver(shared_array, size_of_normal_vec);
}
int main()
{
    // float price_asian = MC_simulation_driver(0.082, 30, 100, 0.1, 0.04, 0.04, 100);
    // cout << price_asian << endl;
    MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // int data_size = 20;
    int NUM_OF_SIM = 1000000;
    vector<float> price_asian = MC_simulation_driver_MPI(0.082, 30, 100, 0.1, 0.04, 0.04, 100, NUM_OF_SIM);
    if (rank == 0)
    {
        float expected_price = price_asian[NUM_OF_SIM];
        cout << expected_price / NUM_OF_SIM << endl;
    }
    MPI_Finalize();
}