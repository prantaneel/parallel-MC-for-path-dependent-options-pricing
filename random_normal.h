#include <random>
#include <math.h>
#include <chrono>

class UniformGenerator
{
private:
public:
    // UniformGenerator();

    // Initialisation and setting the seed
    virtual void init(long seed) = 0;

    // Getting random structures
    virtual double getUniform() = 0;
};

class CustomRandGenerator : public UniformGenerator
{

private:

    std::mt19937 mt_obj;
    std::uniform_real_distribution<> dis_func_obj;

public:
    // CustomRandGenerator();
    void init(long seed)
    {
        std::random_device rd{};
        std::seed_seq ss{rd(), rd(), rd(), rd(),rd(), rd(), rd(), rd()};
        //can we store these seeds to reproduce the results at a later stage?
        
        std::mt19937 mt{ss};
        //create a uniform real distribution
        std::uniform_real_distribution<> dis(0.0, 1.0);

        mt_obj = mt;
        dis_func_obj = dis;
    }

    double getUniform()
    {
        // implementing using a mersenne twister
        double uniform_value_0_1 = dis_func_obj(mt_obj);
        return uniform_value_0_1;
    }
};

class NormalGenerator
{
protected:
    UniformGenerator* ug;
public:
    // NormalGenerator(UniformGenerator& uniformGen);
    virtual double getNormal() = 0;
};

class BoxMuller : public NormalGenerator
{
private:
    double U1, U2;
    double N1, N2;
    double W;
    const double tpi = 2*M_PI;
public:
    BoxMuller(UniformGenerator& uniformGen)
    {
        ug = &uniformGen;
    }
    double getNormal(){
        U1 = ug->getUniform();
        U2 = ug->getUniform();

        W = sqrt(-2.0 * log(U1));

        N1 = W*cos(tpi*U2);
        N2 = W*sin(tpi*U2);

        return N1;
    }

};
