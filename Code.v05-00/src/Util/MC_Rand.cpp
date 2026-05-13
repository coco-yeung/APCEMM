/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/*                                                                  */
/*     Aircraft Plume Chemistry, Emission and Microphysics Model    */
/*                             (APCEMM)                             */
/*                                                                  */
/* MC_Rand Program File                                             */
/*                                                                  */
/* Author               : Thibaud M. Fritz                          */
/* Time                 : 1/25/2019                                 */
/* File                 : MC_Rand.cpp                               */
/*                                                                  */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
#include <iostream>
#include "APCEMM.h"
#include "Util/MC_Rand.hpp"
#include "Core/Input_Mod.hpp"

void setSeed(const OptInput& input) {

    // Sets seed for pseudo-random generator.
    #ifdef DEBUG
        // With DEBUG compile flag set a constant seed for reproducibility
        std::cout << "Compiled in DEBUG mode: random seed is set to 0 for all simulations" << std::endl;
        rng.seed(0 + omp_get_thread_num());
    #else
        if(input.SIMULATION_FORCE_SEED){
            rng.seed(input.SIMULATION_SEED_VALUE + omp_get_thread_num());
            std::cout << "Random seed is set to " << input.SIMULATION_SEED_VALUE << std::endl;
        }
        else{
            // If the seed is not being forced to a value use random device
            rng.seed(std::random_device{}());
        }

    #endif

} /* End of setSeed */

template <typename T>
T fRand(const T fMin, const T fMax) {

    /* Returns a random number between fMin and fMax */

    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(fMin, fMax);
    return (T) dist(rng);

} /* End of fRand */

template double fRand(const double fMin, const double fMax);
template float fRand(const float fMin, const float fMax);
template int fRand(const int fMin, const int fMax);
template unsigned int fRand(const unsigned int fMin, const unsigned int fMax);

/* End of MC_Rand.cpp */
