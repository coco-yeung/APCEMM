#include <FVM_ANDS/AdvDiffSystem.hpp>
#include <chrono>
#include <math.h>
#include <iostream>
#include "APCEMM.h"

namespace FVM_ANDS{
    AdvDiffSystem::AdvDiffSystem(const AdvDiffParams& params, const Vector_1D xCoords, const Vector_1D yCoords, const BoundaryConditions& bc, const Eigen::VectorXd& phi_init, vecFormat format) :
        format_(format),
        u_double_ (params.u),
        v_double_ (params.v),
        shear_ (params.shear),
        dt_ (params.dt),
        dx_ (xCoords[1] - xCoords[0]),
        dy_ (yCoords[1] - yCoords[0]),
        nx_ (xCoords.size()),
        ny_ (yCoords.size()),
        yCoord_(yCoords),
        bcType_top_ (bc.bcType_top),
        bcType_left_ (bc.bcType_left),
        bcType_right_ (bc.bcType_right),
        bcType_bot_ (bc.bcType_bot),
        bcVals_top_ (bc.bcVals_top),
        bcVals_left_ (bc.bcVals_left),
        bcVals_right_ (bc.bcVals_right),
        bcVals_bot_ (bc.bcVals_bot),
        phi_(phi_init)
    {
        invdx_ = 1.0/dx_;
        invdy_ = 1.0/dy_;
        nInteriorPoints_ = nx_ * ny_;
        nGhostPoints_ = 2*nx_ + 2*ny_;
        nTotalPoints_ = nInteriorPoints_ + nGhostPoints_;

        u_vec_.resize(nInteriorPoints_);
        v_vec_.resize(nInteriorPoints_);
        Dh_vec_.resize(nInteriorPoints_);
        Dv_vec_.resize(nInteriorPoints_);
        rhs_.resize(nTotalPoints_);
        phi_.resize(nTotalPoints_);
        points_.reserve(nTotalPoints_);
        deferredCorr_.resize(nInteriorPoints_);
        deferredCorr_.setZero();
        source_.resize(nInteriorPoints_);
        source_.setZero();
        std::generate_n(std::back_inserter(points_), nTotalPoints_, [] { return std::make_unique<Point>(); });
        totalCoefMatrix_.resize(nTotalPoints_, nTotalPoints_);

        #ifdef ENABLE_TIMING
        auto start = std::chrono::high_resolution_clock::now();
        #endif
        updateDiffusion(params.Dh, params.Dv);
        #ifdef ENABLE_TIMING
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "          AdvDiffSys Construtor: updateDiff " << duration_us.count() << " us" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        #endif
        initVelocVecs();
        #ifdef ENABLE_TIMING
        end = std::chrono::high_resolution_clock::now();
        duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "          AdvDiffSys Construtor: initVelocVecs " << duration_us.count() << " us" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        #endif
        buildPointList();
        #ifdef ENABLE_TIMING
        end = std::chrono::high_resolution_clock::now();
        duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "          AdvDiffSys Construtor: buildPointList " << duration_us.count() << " us" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        #endif
        applyBoundaryCondition();
        #ifdef ENABLE_TIMING
        end = std::chrono::high_resolution_clock::now();
        duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "          AdvDiffSys Construtor: applyBoundaryCondition " << duration_us.count() << " us" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        #endif

        buildPointCache();
    }

    void AdvDiffSystem::initVelocVecs(){
        for(int i = 0; i < nx_; i++){
            for(int j = 0; j < ny_; j++){
                int vector_idx = twoDIdx_to_vecIdx(i, j, nx_, ny_, format_);
                u_vec_[vector_idx] = u_double_ - yCoord_[j] * shear_;
                v_vec_[vector_idx] = v_double_;
            }
        }
    }

    void AdvDiffSystem::buildPointList(){
        //interior points
        for(int i = 0; i < nx_; i++){
            for(int j = 0; j < ny_; j++){
                int vector_idx = twoDIdx_to_vecIdx(i, j, nx_, ny_, format_);
                //top interior bound / ghost points
                if(j == ny_ - 1){
                    BoundaryCondDescription bc_ghost = BoundaryCondDescription(bcType_top_, FaceDirection::NORTH, bcVals_top_[i], vector_idx);
                    int corrGhostPoint = nInteriorPoints_ + i;
                    points_[corrGhostPoint] = std::make_unique<GhostPoint>(bc_ghost);

                    BoundaryCondDescription bc_top = BoundaryCondDescription(bcType_top_, FaceDirection::NORTH, bcVals_top_[i], corrGhostPoint);

                    //check for top corner point edge cases
                    //top left
                    if(i == 0){
                        int corrGhostPoint2 = nInteriorPoints_ + nx_ + j;
                        BoundaryCondDescription bc_left = BoundaryCondDescription(bcType_left_, FaceDirection::WEST, bcVals_left_[j], corrGhostPoint2);
                        BoundaryCondDescription bc_ghost2 = BoundaryCondDescription(bcType_left_, FaceDirection::WEST, bcVals_left_[j], vector_idx);
                        points_[vector_idx] = std::make_unique<IntBoundPoint>(bc_top, bc_left);
                        points_[corrGhostPoint2] = std::make_unique<GhostPoint>(bc_ghost2);
                    }
                    //top right
                    else if(i == nx_ - 1){
                        int corrGhostPoint2 = nInteriorPoints_ + nx_ + ny_ + j;
                        BoundaryCondDescription bc_right = BoundaryCondDescription(bcType_right_, FaceDirection::EAST, bcVals_right_[j], corrGhostPoint2);
                        BoundaryCondDescription bc_ghost2 = BoundaryCondDescription(bcType_right_, FaceDirection::EAST, bcVals_right_[j], vector_idx);
                        points_[vector_idx] = std::make_unique<IntBoundPoint>(bc_top, bc_right);
                        points_[corrGhostPoint2] = std::make_unique<GhostPoint>(bc_ghost2);
                    }
                    //if not corner boundary make normal interior node.
                    else{
                        points_[vector_idx] = std::make_unique<IntBoundPoint>(bc_top);
                    }
                }
                //bottom 
                else if (j == 0){
                    int corrGhostPoint = nInteriorPoints_ + nx_ + 2*ny_ + i;
                    BoundaryCondDescription bc_ghost = BoundaryCondDescription(bcType_bot_, FaceDirection::SOUTH, bcVals_bot_[i], vector_idx);
                    points_[corrGhostPoint] = std::make_unique<GhostPoint>(bc_ghost);

                    BoundaryCondDescription bc_bot = BoundaryCondDescription(bcType_bot_, FaceDirection::SOUTH, bcVals_bot_[i], corrGhostPoint);

                    //check for bottom edge cases
                    //bot left
                    if(i == 0){
                        int corrGhostPoint2 = nInteriorPoints_ + nx_ + j;

                        BoundaryCondDescription bc_left = BoundaryCondDescription(bcType_left_, FaceDirection::WEST, bcVals_left_[j], corrGhostPoint2);
                        BoundaryCondDescription bc_ghost2 = BoundaryCondDescription(bcType_left_, FaceDirection::WEST, bcVals_left_[j], vector_idx);
                        
                        points_[vector_idx] = std::make_unique<IntBoundPoint>(bc_bot, bc_left);
                        points_[corrGhostPoint2] = std::make_unique<GhostPoint>(bc_ghost2);
                    }
                    //bot right
                    else if(i == nx_ - 1){
                        int corrGhostPoint2 = nInteriorPoints_ + nx_ + ny_ + j;
                        BoundaryCondDescription bc_right = BoundaryCondDescription(bcType_right_, FaceDirection::EAST, bcVals_right_[j], corrGhostPoint2);
                        BoundaryCondDescription bc_ghost2 = BoundaryCondDescription(bcType_right_, FaceDirection::EAST, bcVals_right_[j], vector_idx);
                        
                        points_[vector_idx] = std::make_unique<IntBoundPoint>(bc_bot, bc_right);
                        points_[corrGhostPoint2] = std::make_unique<GhostPoint>(bc_ghost2);

                    }
                    //if not corner boundary make normal interior node
                    else {
                        points_[vector_idx] = std::make_unique<IntBoundPoint>(bc_bot);
                    }

                }
                // left
                else if (i == 0){
                    int corrGhostPoint = nInteriorPoints_ + nx_ + j;
                    BoundaryCondDescription bc_left = BoundaryCondDescription(bcType_left_, FaceDirection::WEST, bcVals_left_[j], corrGhostPoint);
                    BoundaryCondDescription bc_ghost = BoundaryCondDescription(bcType_left_, FaceDirection::WEST, bcVals_left_[j], vector_idx);

                    points_[vector_idx] = std::make_unique<IntBoundPoint>(bc_left);
                    points_[corrGhostPoint] = std::make_unique<GhostPoint>(bc_ghost);
                } 
                //right 
                else if (i == nx_ - 1){
                    int corrGhostPoint = nInteriorPoints_ + nx_ + ny_ + j;
                    BoundaryCondDescription bc_right = BoundaryCondDescription(bcType_right_, FaceDirection::EAST, bcVals_right_[j], corrGhostPoint);
                    BoundaryCondDescription bc_ghost = BoundaryCondDescription(bcType_right_, FaceDirection::EAST, bcVals_right_[j], vector_idx);

                    points_[vector_idx] = std::make_unique<IntBoundPoint>(bc_right);
                    points_[corrGhostPoint] = std::make_unique<GhostPoint>(bc_ghost);
                }
                else {
                    points_[vector_idx] = std::make_unique<Point>(BoundaryConditionFlag::INTERIOR);
                }
            }
        }
    }
    void AdvDiffSystem::buildCoeffMatrix(bool operatorSplit){
        // Skip if we have a prebuilt matrix
        if (use_shared_totalCoefMatrix_) {
            return;
        }
        
        //Crank-Nicholson Discretization. Builds the Advection terms of the A matrix 
        //in the system A * phi_t+1 = b.

        //Num non-zeros calculation: 
        std::vector<Eigen::Triplet<double>> tripletList;
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < nTotalPoints_; i++){

            if(points_[i]->isGhost()){
                switch(points_[i]->bcType()){
                    case BoundaryConditionFlag::DIRICHLET_GHOSTPOINT:{
                        // (phi_int + phi_ghost) / 2 = phi_boundary
                        // if inhomog, the bc value will appear in the rhs.
                        tripletList.emplace_back(i, i, 0.5);
                        tripletList.emplace_back(i, points_[i]->corrPoint(), 0.5);
                        break;
                    }
                    case BoundaryConditionFlag::PERIODIC_GHOSTPOINT:{
                        // Ghost node will exactly equal to the cell on the other end of the domain
                        // effectively makes opposite ends of the domain neighbor points.
                        throw std::runtime_error("Periodic BCs not yet implemented.");
                        break;
                    }
                    default: {
                        throw std::runtime_error("Ghost point doesn't have a bcType associated with being a ghost point!");
                    }
                }
                continue;
            }

            int idx_E = neighbor_point(FaceDirection::EAST, i);
            int idx_W = neighbor_point(FaceDirection::WEST, i);
            int idx_N = neighbor_point(FaceDirection::NORTH, i);
            int idx_S = neighbor_point(FaceDirection::SOUTH, i);

            //Diffusion Terms
            double coeff_C = 1 + 2 * dt_ * (Dh_vec_[i] / (dx_ * dx_) + Dv_vec_[i] / (dy_ * dy_));
            double coeff_E = -Dh_vec_[i] * dt_ / (dx_ * dx_);
            double coeff_W = -Dh_vec_[i] * dt_ / (dx_ * dx_) ;
            double coeff_N = -Dv_vec_[i] * dt_ / (dy_ * dy_);
            double coeff_S = -Dv_vec_[i] * dt_ / (dy_ * dy_);

            //Operator splitting uses implicit only for diffusion
            if(!operatorSplit && (u_double_ > 0 || v_double_ > 0 || shear_ > 0)){
                buildAdvectionCoeffs(i, coeff_C, coeff_N, coeff_S, coeff_E, coeff_W);
            }

            //Triplet Format: row, col, value
            tripletList.emplace_back(i, idx_E, coeff_E);
            tripletList.emplace_back(i, idx_W, coeff_W);
            tripletList.emplace_back(i, idx_N, coeff_N);
            tripletList.emplace_back(i, idx_S, coeff_S);
            tripletList.emplace_back(i, i, coeff_C);
        } 

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
        totalCoefMatrix_.setFromTriplets(tripletList.begin(), tripletList.end());
    }
    void AdvDiffSystem::buildAdvectionCoeffs(int i, double& coeff_C, double& coeff_N, double& coeff_S, double& coeff_E, double& coeff_W){
        //Advection Terms
        //When a boundary condition is in place, phi at the face can be directly calculated using the BC.
        //Therefore, that term goes to the RHS and the contribution of that face to the coeffs goes to 0.

        bool isNorthBoundary = 0, isWestBoundary = 0, isEastBoundary = 0, isSouthBoundary = 0;
        if(points_[i]->bcType() != BoundaryConditionFlag::INTERIOR){
            isNorthBoundary = points_[i]->bcDirection() == FaceDirection::NORTH;
            isSouthBoundary = points_[i]->bcDirection() == FaceDirection::SOUTH;

            //Corner cases...
            bool secondaryWestBound = (points_[i]->secondBoundaryConds() && points_[i]->secondBoundaryConds().value().direction == FaceDirection::WEST);
            bool secondaryEastBound = (points_[i]->secondBoundaryConds() && points_[i]->secondBoundaryConds().value().direction == FaceDirection::EAST);

            isWestBoundary = (points_[i]->bcDirection() == FaceDirection::WEST || secondaryWestBound);
            isEastBoundary = (points_[i]->bcDirection() == FaceDirection::EAST || secondaryEastBound);
        }

        int idx_E = neighbor_point(FaceDirection::EAST, i);
        int idx_W = neighbor_point(FaceDirection::WEST, i);
        int idx_N = neighbor_point(FaceDirection::NORTH, i);
        int idx_S = neighbor_point(FaceDirection::SOUTH, i);

        double u_W = isWestBoundary? u_vec_[i] : 0.5 * (u_vec_[i] + u_vec_[idx_W]);
        double u_E = isEastBoundary? u_vec_[i] : 0.5 * (u_vec_[i] + u_vec_[idx_E]);
        double v_N = isNorthBoundary? v_vec_[i] : 0.5 * (v_vec_[i] + v_vec_[idx_N]);
        double v_S = isSouthBoundary? v_vec_[i] : 0.5 * (v_vec_[i] + v_vec_[idx_S]);

        if (u_E >= 0) coeff_C += u_E * dt_ / dx_;
        if (u_W < 0) coeff_C -= u_W * dt_ / dx_;
        if (v_N >= 0) coeff_C += v_N * dt_ / dy_;
        if (v_S < 0) coeff_C -= v_S * dt_ / dy_;

        if (u_E < 0) coeff_E += u_E * dt_ / dy_;

        if (u_W >= 0) coeff_W -= u_W * dt_ / dx_;

        if (v_N < 0) coeff_N += v_N * dt_ / dy_;

        if (v_S >= 0) coeff_S -= v_S * dt_ / dy_;

        double phi_N_corr = 0, phi_S_corr = 0, phi_W_corr = 0, phi_E_corr = 0;

        auto bool_to_signed = [](bool binary) { return binary ? 1 : -1; };

        if (!isNorthBoundary)
            phi_N_corr -= bool_to_signed(v_N >= 0) * v_N * dt_ / dy_ * 0.5 * minmod(i, FaceDirection::NORTH, 0) * (phi_[idx_N] - phi_[i]);

        if (!isSouthBoundary)
            phi_S_corr += bool_to_signed(v_S >= 0) * v_S * dt_ / dy_ * 0.5 * minmod(i, FaceDirection::SOUTH, 0) * (phi_[i] - phi_[idx_S]);
        
        if (!isWestBoundary)
            phi_W_corr += bool_to_signed(u_W >= 0) * u_W * dt_ / dx_ * 0.5 * minmod(i, FaceDirection::WEST, 0) * (phi_[i] - phi_[idx_W]);

        if (!isEastBoundary)
            phi_E_corr -= bool_to_signed(u_E >= 0) * u_E * dt_ / dx_ * 0.5 * minmod(i, FaceDirection::EAST, 0) * (phi_[idx_E] - phi_[i]);
        
        double TVD_deferred_corr = phi_N_corr + phi_S_corr + phi_W_corr + phi_E_corr;
        deferredCorr_[i] = TVD_deferred_corr;
    }

    const Eigen::VectorXd& AdvDiffSystem::calcRHS(){
        for(int i = 0; i < nTotalPoints_; i++){
            if(points_[i]->isGhost()){
                switch(points_[i]->bcType()){
                    case BoundaryConditionFlag::DIRICHLET_GHOSTPOINT:{
                        // Equation: (phi_int + phi_ghost) / 2 = phi_boundary
                        rhs_[i] = points_[i]->bcVal();
                        break;
                    }
                    case BoundaryConditionFlag::PERIODIC_GHOSTPOINT:{
                        // Ghost node will exactly equal to the cell on the other end of the domain
                        // effectively makes opposite ends of the domain neighbor points.
                        throw std::runtime_error("Periodic BCs not yet implemented.");
                        break;
                    }
                    default: {
                        throw std::runtime_error("Ghost point doesn't have a bcType associated with being a ghost point!");
                    }                
                }
                continue;
            }

            switch(points_[i]->bcType()){
                case BoundaryConditionFlag::INTERIOR:{
                    rhs_[i] = phi_[i] + deferredCorr_[i] + source_[i]*dt_;
                    break;
                }
                case BoundaryConditionFlag::DIRICHLET_INT_BPOINT:{
                    rhs_[i] = phi_[i] + deferredCorr_[i] + source_[i]*dt_;
                    switch(points_[i]->bcDirection()){
                        case FaceDirection::NORTH:
                            rhs_[i] -= v_vec_[i] * dt_ / dy_ * points_[i]->bcVal();
                            break;
                        case FaceDirection::SOUTH:
                            rhs_[i] += v_vec_[i] * dt_ / dy_ * points_[i]->bcVal();
                            break;
                        case FaceDirection::EAST:
                            rhs_[i] -= u_vec_[i] * dt_ / dx_ * points_[i]->bcVal();
                            break;
                        case FaceDirection::WEST:
                            rhs_[i] += u_vec_[i] * dt_ / dx_ * points_[i]->bcVal();
                            break;
                        case FaceDirection::ERROR:
                            throw std::runtime_error("Invalid FaceDirection in Dirichlet boundary condition");
                    }
                    if (!points_[i]->secondBoundaryConds()) break;
                    BoundaryCondDescription bc_2 = points_[i]->secondBoundaryConds().value();
                    switch(bc_2.direction){
                        case FaceDirection::EAST:
                            rhs_[i] -= u_vec_[i] * dt_ / dx_ * bc_2.bcVal;
                            break;
                        case FaceDirection::WEST:
                            rhs_[i] += u_vec_[i] * dt_ / dx_ * bc_2.bcVal;
                            break;
                        default:
                            throw std::runtime_error("Can't have anything but EAST or WEST as secondary BC!");
                    }
                    break;
                }
                case BoundaryConditionFlag::PERIODIC_INT_BPOINT:{
                    throw std::runtime_error("Periodic BCs not yet implemented.");
                    break;
                }
                default: {
                    throw std::runtime_error("Interior boundary point has invalid bcType");
                }                
            }
            continue;
        }

        return rhs_;
    }
    void AdvDiffSystem::applyBoundaryCondition(){
        //top and bottom bc
        for(int i = 0; i < nx_; i++){
            //top
            int bPointID_top = twoDIdx_to_vecIdx(i, ny_ - 1, nx_, ny_, format_);
            //: int twoDIdx_to_vecIdx(int idx_x, int idx_y, int nx, int ny, vecFormat format){
            //     return (format == vecFormat::ROWMAJOR) ?
            //             idx_y * nx + idx_x :
            //             idx_x * ny + idx_y;
            // }
            switch(bcType_top_){
                case BoundaryConditionFlag::DIRICHLET_INT_BPOINT: {
                    int ghostPointID = points_[bPointID_top]->corrPoint();
                    phi_[ghostPointID] = 2 * points_[bPointID_top]->bcVal() - phi_[bPointID_top];
                    break;
                }
                default: {
                    throw std::runtime_error("Chosen boundary condition not implemented yet");
                }
            }

            //bottom
            int bPointID_bot = twoDIdx_to_vecIdx(i, 0, nx_, ny_, format_);
            switch(bcType_bot_){
                case BoundaryConditionFlag::DIRICHLET_INT_BPOINT: {
                    int ghostPointID = points_[bPointID_bot]->corrPoint();
                    phi_[ghostPointID] = 2 * points_[bPointID_bot]->bcVal() - phi_[bPointID_bot];
                    break;
                }
                default: {
                    throw std::runtime_error("Chosen boundary condition not implemented yet");
                }
            }
        }

        //left and right bc
        for(int j = 0; j < ny_; j++){
            //corner cases
            if(j == 0 || j == ny_ - 1){
                int bPointID_cornerLeft = twoDIdx_to_vecIdx(0, j, nx_, ny_, format_);
                int ghostPointID = points_[bPointID_cornerLeft]->secondBoundaryConds().value().corrPoint;
                double bcVal =  points_[bPointID_cornerLeft]->secondBoundaryConds().value().bcVal;
                switch(bcType_left_){
                    case BoundaryConditionFlag::DIRICHLET_INT_BPOINT: {
                        phi_[ghostPointID] = 2 * bcVal - phi_[bPointID_cornerLeft];
                        break;
                    }
                    default: {
                        throw std::runtime_error("Chosen boundary condition not implemented yet");
                    }
                }

                int bPointID_cornerRight = twoDIdx_to_vecIdx(nx_ - 1, j, nx_, ny_, format_);
                ghostPointID = points_[bPointID_cornerRight]->secondBoundaryConds().value().corrPoint;
                bcVal = points_[bPointID_cornerRight]->secondBoundaryConds().value().bcVal;

                switch(bcType_right_){
                    case BoundaryConditionFlag::DIRICHLET_INT_BPOINT: {
                        phi_[ghostPointID] = 2 * bcVal - phi_[bPointID_cornerRight];
                        break;
                    }
                    default: {
                        throw std::runtime_error("Chosen boundary condition not implemented yet");
                    }
                }
            }

            //left
            int bPointID_left = twoDIdx_to_vecIdx(0, j, nx_, ny_, format_);
            switch(bcType_left_){
                case BoundaryConditionFlag::DIRICHLET_INT_BPOINT: {
                    int ghostPointID = points_[bPointID_left]->corrPoint();
                    phi_[ghostPointID] = 2 * points_[bPointID_left]->bcVal() - phi_[bPointID_left];
                    break;
                }
                default: {
                    throw std::runtime_error("Chosen boundary condition not implemented yet");
                }
            }
            //right
            int bPointID_right = twoDIdx_to_vecIdx(nx_ - 1, j, nx_, ny_, format_);
            switch(bcType_right_){
                case BoundaryConditionFlag::DIRICHLET_INT_BPOINT: {
                    int ghostPointID = points_[bPointID_right]->corrPoint();
                    phi_[ghostPointID] = 2 * points_[bPointID_right]->bcVal() - phi_[bPointID_right];
                    break;
                }
                default: {
                    throw std::runtime_error("Chosen boundary condition not implemented yet");
                }
            }
        }
    }
    void AdvDiffSystem::updateBoundaryCondition(const BoundaryConditions& bc){
        bcType_top_ = bc.bcType_top;
        bcType_left_ = bc.bcType_left;
        bcType_right_ = bc.bcType_right;
        bcType_bot_ = bc.bcType_bot;
        bcVals_top_ = bc.bcVals_top;
        bcVals_left_ = bc.bcVals_left;
        bcVals_right_ = bc.bcVals_right;
        bcVals_bot_ = bc.bcVals_bot;

        //Go through ghost points, and update the bcType and value of them and their corresponding interior nodes
        //As seen in buildPointList(), ghost point order goes top->left->right->bottom
        int currIdx = nInteriorPoints_;
        //top
        for(int i = 0; i < nx_; i++){
            int corrPointID = points_[currIdx]->corrPoint();
            points_[currIdx]->setBCType(bcType_top_); 
            points_[currIdx]->setBCVal(bcVals_top_[i]); 
            points_[corrPointID]->setBCType(bcType_top_); 
            points_[corrPointID]->setBCVal(bcVals_top_[i]); 
            currIdx++;
        }
        //left
        for(int i = 0; i < ny_; i++){
            int corrPointID = points_[currIdx]->corrPoint();
            if(i == 0 || i == ny_ - 1){
                points_[currIdx]->setBCType(bcType_left_); 
                points_[currIdx]->setBCVal(bcVals_left_[i]);
                BoundaryCondDescription bc(bcType_left_, FaceDirection::WEST, bcVals_left_[i], currIdx);
                points_[corrPointID]->setSecondaryBC(bc);
                currIdx++;
                continue;
            }
            points_[currIdx]->setBCType(bcType_left_); 
            points_[currIdx]->setBCVal(bcVals_left_[i]); 
            points_[corrPointID]->setBCType(bcType_left_); 
            points_[corrPointID]->setBCVal(bcVals_left_[i]);
            currIdx++;
        }
        //right
        for(int i = 0; i < ny_; i++){
            int corrPointID = points_[currIdx]->corrPoint();
            if(i == 0 || i == ny_ - 1){
                points_[currIdx]->setBCType(bcType_right_); 
                points_[currIdx]->setBCVal(bcVals_right_[i]);
                BoundaryCondDescription bc(bcType_right_, FaceDirection::EAST, bcVals_right_[i], currIdx);
                points_[corrPointID]->setSecondaryBC(bc);
                currIdx++;
                continue;
            }
            points_[currIdx]->setBCType(bcType_right_); 
            points_[currIdx]->setBCVal(bcVals_right_[i]); 
            points_[corrPointID]->setBCType(bcType_right_); 
            points_[corrPointID]->setBCVal(bcVals_right_[i]);
            currIdx++;
        }
        //bot
        for(int i = 0; i < nx_; i++){
            int corrPointID = points_[currIdx]->corrPoint();
            points_[currIdx]->setBCType(bcType_bot_); 
            points_[currIdx]->setBCVal(bcVals_bot_[i]); 
            points_[corrPointID]->setBCType(bcType_bot_); 
            points_[corrPointID]->setBCVal(bcVals_bot_[i]); 
            currIdx++;
        }
        applyBoundaryCondition(); //need this to calculate minmod function at some timestep.
    }

    void AdvDiffSystem::buildPointCache() {
        // previously contained in forwardEulerAdvection
        // build once since position of boundary and interior indices do not change
        // if size of nx_ and ny_ do not change
        interiorIndices_.clear();
        boundaryIndices_.clear();
        pointCache_.clear();
        pointCache_.resize(nInteriorPoints_);
        
        for(int i = 0; i < nInteriorPoints_; i++){
            //When a boundary condition is in place, phi at the face can be directly calculated using the BC.
            //Therefore, that term goes to the RHS and the contribution of that face to the coeffs goes to 0.
            bool isNorthBoundary = 0, isWestBoundary = 0, isEastBoundary = 0, isSouthBoundary = 0, secondaryEastBound = 0, secondaryWestBound = 0;
            int idx_E = i + ny_;
            int idx_W = i - ny_;
            int idx_N = i + 1;
            int idx_S = i - 1;
            double bcVal = 0.0, secondaryBcVal = 0.0;

            //commenting out this results in ~30% speedup
            //The calls involving the optional are maybe 1/3 of the cost. Maybe something to look at later.
            if(points_[i]->bcType() != BoundaryConditionFlag::INTERIOR
            || !isValidPointID(i+2) || !isValidPointID(i-2)
            || !isValidPointID(i+2*ny_) || !isValidPointID(i-2*ny_)){
                Point* point = points_[i].get();
                FaceDirection direction = point->bcDirection();
                isNorthBoundary = direction == FaceDirection::NORTH;
                isSouthBoundary = direction == FaceDirection::SOUTH;

                //Corner cases...
                secondaryWestBound = (point->secondBoundaryConds() && point->secondBoundaryConds()->direction == FaceDirection::WEST);
                secondaryEastBound = (point->secondBoundaryConds() && point->secondBoundaryConds()->direction == FaceDirection::EAST);

                isWestBoundary = (direction == FaceDirection::WEST || secondaryWestBound);
                isEastBoundary = (direction == FaceDirection::EAST || secondaryEastBound);

                //only call this lookup function on boundary nodes which are inconsequential in number
                idx_N = isNorthBoundary? point->corrPoint() : idx_N;
                idx_S = isSouthBoundary? point->corrPoint() : idx_S;
                idx_E = isEastBoundary? (secondaryEastBound ? point->secondBoundaryConds()->corrPoint : point->corrPoint()) : idx_E;
                idx_W = isWestBoundary? (secondaryEastBound ? point->secondBoundaryConds()->corrPoint : point->corrPoint()) : idx_W;

                bcVal = point->bcVal();
                secondaryBcVal = secondaryWestBound || secondaryEastBound ? point->secondBoundaryConds()->bcVal : 0.0;

                pointCache_[i] = {
                    isNorthBoundary, isSouthBoundary, isEastBoundary, isWestBoundary,
                    secondaryWestBound, secondaryEastBound,
                    idx_N, idx_S, idx_E, idx_W,
                    bcVal, secondaryBcVal
                };

                boundaryIndices_.push_back(i);
            }
            else {
                interiorIndices_.push_back(i);
            }
        }
    }

    Eigen::VectorXd AdvDiffSystem::semiLagrangianAdvection(bool parallelAdvection) const noexcept{
        Eigen::VectorXd soln(nTotalPoints_);

        for(int i = 0; i < nInteriorPoints_; i++){
            double ix = i / ny_;
            double iy = i % ny_;
            double u_local = u_vec_[i];
            double v_local = v_vec_[i];
            
            // find departure point
            double ix_dep = ix - u_local * dt_ * invdx_;
            double iy_dep = iy - v_local * dt_ * invdy_;

            // if departure point is outside boundary
            if (iy_dep < 0){
                double ix_temp = std::max(0.0, std::min(ix_dep, nx_ - 1.0));
                int ix_1 = static_cast<int>(std::floor(ix_temp));
                int ix_2 = std::min(ix_1 + 1, nx_ - 1);
                double wx = ix_temp - ix_1;
                soln[i] = lerp(bcVals_bot_[ix_1], bcVals_bot_[ix_2], wx) + source_[i] * dt_;
                continue;
            }
            else if (iy_dep > ny_ - 1){
                double ix_temp = std::max(0.0, std::min(ix_dep, nx_ - 1.0));
                int ix_1 = static_cast<int>(std::floor(ix_temp));
                int ix_2 = std::min(ix_1 + 1, nx_ - 1);
                double wx = ix_temp - ix_1;
                soln[i] = lerp(bcVals_top_[ix_1], bcVals_top_[ix_2], wx) + source_[i] * dt_;
                continue;
            }
            else if (ix_dep < 0){
                double iy_temp = std::max(0.0, std::min(iy_dep, ny_ - 1.0));
                int iy_1 = static_cast<int>(std::floor(iy_temp));
                int iy_2 = std::min(iy_1 + 1, ny_ - 1);
                double wy = iy_temp - iy_1;
                soln[i] = lerp(bcVals_left_[iy_1], bcVals_left_[iy_2], wy) + source_[i] * dt_;
                continue;
            }
            else if (ix_dep > nx_ - 1){
                double iy_temp = std::max(0.0, std::min(iy_dep, ny_ - 1.0));
                int iy_1 = static_cast<int>(std::floor(iy_temp));
                int iy_2 = std::min(iy_1 + 1, ny_ - 1);
                double wy = iy_temp - iy_1;
                soln[i] = lerp(bcVals_right_[iy_1], bcVals_right_[iy_2], wy) + source_[i] * dt_;
                continue;
            }

            // find 4 nearest points
            int ix_1 = static_cast<int>(std::floor(ix_dep));
            int iy_1 = static_cast<int>(std::floor(iy_dep));
            int ix_2 = std::min(ix_1 + 1, nx_ - 1);
            int iy_2 = std::min(iy_1 + 1, ny_ - 1);

            // find fractional weight
            double wx = ix_dep - ix_1;
            double wy = iy_dep - iy_1;

            // convert into phi indexing
            ix_1 *= ny_;
            ix_2 *= ny_;

            // find phi values of 4 nearest points
            double phi_11 = phi_[ix_1 + iy_1];
            double phi_12 = phi_[ix_1 + iy_2];
            double phi_21 = phi_[ix_2 + iy_1];
            double phi_22 = phi_[ix_2 + iy_2];


            // linear interpolate horizontal
            double phi_x1 = lerp(phi_11, phi_21, wx);
            double phi_x2 = lerp(phi_12, phi_22, wx);
            // vertical
            soln[i] = lerp(phi_x1, phi_x2, wy) + source_[i] * dt_;

        }
        return soln;
    }

    Eigen::VectorXd AdvDiffSystem::forwardEulerAdvection(bool operatorSplit, bool parallelAdvection) const noexcept{
        Eigen::VectorXd soln(nTotalPoints_);

        // double avgBackgroundCalcTime = 0;
        //Explicit Time-Stepping
        #pragma omp parallel for    \
        if      ( parallelAdvection ) \
        default ( shared          ) \
        schedule( static, 100      )
        for(int i : interiorIndices_){
            double phi_P  = phi_[i];
            double phi_N  = phi_[i + 1];
            double phi_S  = phi_[i - 1];
            double phi_E  = phi_[i + ny_];
            double phi_W  = phi_[i - ny_];
            double phi_NN = phi_[i + 2];
            double phi_SS = phi_[i - 2];
            double phi_EE = phi_[i + 2*ny_];
            double phi_WW = phi_[i - 2*ny_];

            double u_local = u_vec_[i];
            double v_local = v_vec_[i];
            double phi_N_new, phi_S_new, phi_W_new, phi_E_new;

            // select r =  dS if v >=0 else phi_NN - phi_N
            // if either r or dN is negative, return zero
            // else if r > dN, return dN
            // else return r 

            double dN = phi_N - phi_P;
            double dS = phi_P - phi_S;

            if(v_local >= 0){
                double lim_N = minmod_nodiv(dS, dN);
                phi_N_new = phi_P + 0.5 * lim_N;
                double dSS = phi_S - phi_SS;
                double lim_S = minmod_nodiv(dSS, dS);
                phi_S_new = phi_S + 0.5 * lim_S;
            } else {
                double dNN = phi_NN - phi_N;
                double lim_N = minmod_nodiv(dNN, dN);
                phi_N_new = phi_N - 0.5 * lim_N;
                double lim_S = neighbor_point(FaceDirection::NORTH, i) ? 0 : minmod_nodiv(dN, dS);
                phi_S_new = phi_P - 0.5 * lim_S;
            }

            double dE = phi_E - phi_P;
            double dW = phi_P - phi_W;

            if(u_local >= 0){
                double lim_E = minmod_nodiv(dW, dE);
                phi_E_new = phi_P + 0.5 * lim_E;
                double dWW = phi_W - phi_WW;
                double lim_W = minmod_nodiv(dWW, dW);
                phi_W_new = phi_W + 0.5 * lim_W;
            } else {
                double dEE = phi_EE - phi_E;
                double lim_E = minmod_nodiv(dEE, dE);
                phi_E_new = phi_E - 0.5 * lim_E;
                double lim_W = minmod_nodiv(dE, dW);
                phi_W_new = phi_P - 0.5 * lim_W;
            }

            soln[i] = dt_ * invdx_ * (u_local * phi_W_new - u_local * phi_E_new)
                    + dt_ * invdy_ * (v_local * phi_S_new - v_local * phi_N_new)
                    + source_[i] * dt_ + phi_P;
        }

        // double avgBackgroundCalcTime = 0;
        //Explicit Time-Stepping
        #pragma omp parallel for    \
        if      ( parallelAdvection ) \
        default ( shared          ) \
        schedule( static, 100      )
        for(int i : boundaryIndices_){
            const PointCache& pc = pointCache_[i];

            double u_local = u_vec_[i];
            double v_local = v_vec_[i];

            double phi_N, phi_S, phi_W, phi_E;

            if(pc.isNorth){
                phi_N = pc.bcVal;
            }
            else if (v_local >= 0){
                phi_N = phi_[i] + 0.5 * minmod_N_vPos(i) * (phi_[pc.idx_N] - phi_[i]);
            }
            else {
                phi_N = phi_[pc.idx_N] + 0.5 * minmod_N_vNeg(i) * (phi_[i] - phi_[pc.idx_N]);
            }
            if(pc.isSouth){
                phi_S = pc.bcVal;
            }
            else if (v_local >= 0){
                phi_S = phi_[pc.idx_S] +  0.5 * minmod_S_vPos(i) * (phi_[i] - phi_[pc.idx_S]);
            }
            else {
                phi_S = phi_[i] +  0.5 * minmod_S_vNeg(i) * (phi_[pc.idx_S] - phi_[i]);
            }

            if(pc.isWest){
                phi_W = pc.secondaryWest ? pc.secondaryBcVal : pc.bcVal;
            }
            else if (u_local >= 0){
                phi_W = phi_[pc.idx_W] + 0.5 * minmod_W_vPos(i) * (phi_[i] - phi_[pc.idx_W]);
            }
            else {
                phi_W = phi_[i] + 0.5 * minmod_W_vNeg(i) * (phi_[pc.idx_W] - phi_[i]);
            }

            if(pc.isEast){
                phi_E = pc.secondaryEast ? pc.secondaryBcVal : pc.bcVal;
            }
            else if (u_local >= 0){
                phi_E = phi_[i] + 0.5 * minmod_E_vPos(i) * (phi_[pc.idx_E] - phi_[i]);
            }
            else {
                phi_E = phi_[pc.idx_E] + 0.5 * minmod_E_vNeg(i) * (phi_[i] - phi_[pc.idx_E]);
            }

            //Even just setting this to 0 is like a 2 ns save out of 12, not sure if worth
            soln[i] = /*(!operatorSplit) * (Dh_ * dt_ * invdx_ * (dphi_dx_E - dphi_dx_W) + Dv_ * dt_ * invdy_ * (dphi_dy_N - dphi_dy_S))\*/
                     dt_ * invdx_ * (u_local * phi_W - u_local * phi_E) + dt_ * invdy_ * (v_local * phi_S - v_local * phi_N)\
                    + source_[i] * dt_ + phi_[i];
        }
        return soln;
    }
    
void sor_solve(const Eigen::SparseMatrix<double, Eigen::RowMajor> &A, const Eigen::VectorXd &rhs, Eigen::VectorXd &phi, double omega, double threshold, int n_iters) {
    /*
    diagCoeff should always be overwritten in the for loop
    before we get to "x_i *= omega / diagCoeff;"
    Setting it to 0 instead of leaving uninitialized guarantees
    that we get an error if for some reason diagCoeff is not overwritten
    Not sure how to do this better for now...
    */ 
    double diagCoeff = 0;
    double residual = 1;

    while(residual > threshold){
        const double* valuePtr = A.valuePtr();
        const int* innerIdxPtr = A.innerIndexPtr();
        const int* outerIdxPtr = A.outerIndexPtr();

        for(int iteration = 0; iteration < n_iters; iteration++){
            int outerIdx = 0;

            for (int i = 0; i < rhs.size(); i++) {
                double x_i = 0;
                int rowStartIdx = outerIdxPtr[outerIdx];
                int rowEndIdx = outerIdxPtr[outerIdx + 1];
                for (int j = rowStartIdx; j < rowEndIdx; j++) {

                    if (innerIdxPtr[j] == i) {
                        diagCoeff = valuePtr[j];
                        continue;
                    }
                    x_i -= valuePtr[j] * phi[innerIdxPtr[j]];
                }
                x_i += rhs[i];

                x_i *= omega / diagCoeff;
                x_i += (1 - omega) * phi[i];
                phi[i] = x_i;
                outerIdx++;
            } // end inner for loop
        } // end iters for loop
    
        residual = (A * phi - rhs).eval().lpNorm<2>()/ rhs.lpNorm<2>();
        if (isnan(residual)) throw std::runtime_error("NaN residual encountered");
    } // end while loop

}

}