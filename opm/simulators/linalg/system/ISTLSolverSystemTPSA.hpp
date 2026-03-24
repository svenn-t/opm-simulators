#pragma once

#include "SystemPreconditionerFactoryTPSA.hpp"
#include "SystemTypes.hpp"
#include "MatrixResidualSplitterTPSA.hpp"

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/ISTLSolverTPSA.hpp>

namespace Opm
{

template <class TypeTag>
class ISTLSolverSystemTPSA : public ISTLSolverTPSA<TypeTag>
{
protected:
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using SparseMatrixAdapter = GetPropType<TypeTag, Properties::SparseMatrixAdapterTPSA>;
    using Vector = GetPropType<TypeTag, Properties::GlobalEqVectorTPSA>;
    using Indices = GetPropType<TypeTag, Properties::IndicesTPSA>;
    using WellModel = GetPropType<TypeTag, Properties::WellModel>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using Matrix = typename SparseMatrixAdapter::IstlMatrix;
    using ThreadManager = GetPropType<TypeTag, Properties::ThreadManager>;
    using ElementContext = GetPropType<TypeTag, Properties::ElementContext>;
    using AbstractSolverType = Dune::InverseOperator<Vector, Vector>;
    using AbstractOperatorType = Dune::AssembledLinearOperator<Matrix, Vector, Vector>;
    using AbstractPreconditionerType = Dune::PreconditionerWithUpdate<Vector, Vector>;
    using ElementMapper = GetPropType<TypeTag, Properties::ElementMapper>;
    using ElementChunksType = ElementChunks<GridView, Dune::Partitions::All>;

    using Splitter = MatrixResidualSplitterTPSA<Scalar, Matrix, Vector>;

    constexpr static std::size_t pressureIndex = 0;

#if HAVE_MPI
    using CommunicationType = Dune::OwnerOverlapCopyCommunication<int, int>;
#else
    using CommunicationType = Dune::Communication<int>;
#endif

    using Parent = ISTLSolverTPSA<TypeTag>;

    static constexpr auto _0 = Dune::Indices::_0;
    static constexpr auto _1 = Dune::Indices::_1;
    static constexpr auto _2 = Dune::Indices::_2;

public:
    ISTLSolverSystemTPSA(const Simulator& simulator,
                     const FlowLinearSolverParameters& parameters,
                     bool forceSerial = false)
        : Parent(simulator, parameters, forceSerial)
    {
    }

    explicit ISTLSolverSystemTPSA(const Simulator& simulator)
        : Parent(simulator)
    {
    }

    void prepare(const SparseMatrixAdapter& M, Vector& b) override
    {
        OPM_TIMEBLOCK(istlSolverPrepare);
        this->initPrepare(M.istlMatrix(), b);
        prepareSystemSolver();
    }

    void prepare(const Matrix& M, Vector& b) override
    {
        OPM_TIMEBLOCK(istlSolverPrepare);
        this->initPrepare(M, b);
        prepareSystemSolver();
    }

    bool solve(Vector& x) override
    {
        OPM_TIMEBLOCK(istlSolverSolve);
        ++this->solveCount_;

        const size_t numCells = this->matrix_->N();
        sysX_[_0].resize(numCells);
        sysX_[_0] = 0.0;
        sysX_[_1].resize(numCells);
        sysX_[_1] = 0.0;
        sysX_[_2].resize(numCells);
        sysX_[_2] = 0.0;

        Dune::InverseOperatorResult result;
        sysSolver_->apply(sysX_, sysRhs_, result);
        this->iterations_ = result.iterations;

        // sysX_ -> x
        postProcessSolution(x);

        return this->checkConvergence(result);
    }

    void setSparsityPattern(const std::vector<std::set<unsigned> >& sparsityPattern)
    {
        sparsityPattern_ = sparsityPattern;
    }

private:
    void prepareSystemSolver()
    {
        OPM_TIMEBLOCK(flexibleSolverPrepare);

        const bool localNeedRebuild = !sysInitialized_;

        // All ranks must agree: rebuild if ANY rank needs it, since
        // create and update take different MPI-collective code paths
        // (AMG hierarchy construction vs. update).
        const bool needRebuild
            = this->comm_->communicator().max(static_cast<int>(localNeedRebuild)) > 0;

        // Split parent matrix and residual into sub-matrices and residuals, one per equation/PV:
        // 3 displacement, 3 rotation, and 1 solid pressure
        Splitter splitter(this->simulator_, *this->matrix_, *this->rhs_, sparsityPattern_);
        splitter.generateSubmatrices();

        // Set sub-matrices in system matrix
        DDmat_ = splitter.takeDispDispMatrix();
        DRmat_ = splitter.takeDispRotMatrix();
        DSmat_ = splitter.takeDispSPresMatrix();
        RDmat_ = splitter.takeRotDispMatrix();
        RRmat_ = splitter.takeRotRotMatrix();
        RSmat_ = splitter.takeRotSPresMatrix();
        SDmat_ = splitter.takeSPresDispMatrix();
        SRmat_ = splitter.takeSPresRotMatrix();
        SSmat_ = splitter.takeSPresSPresMatrix();

        Scalar lam1 = 1.0e-5;
        Scalar lam2 = 1.0e5;

        DDmat_->istlMatrix() *= lam1 * lam1;
        RRmat_->istlMatrix() *= lam2 * lam2;
        SSmat_->istlMatrix() *= lam2 * lam2;

        sysMatrix_.M11 = DDmat_.get();
        sysMatrix_.M12 = DRmat_.get();
        sysMatrix_.M13 = DSmat_.get();
        sysMatrix_.M21 = RDmat_.get();
        sysMatrix_.M22 = RRmat_.get();
        sysMatrix_.M23 = RSmat_.get();
        sysMatrix_.M31 = SDmat_.get();
        sysMatrix_.M32 = SRmat_.get();
        sysMatrix_.M33 = SSmat_.get();

        // Set sub-residuals
        splitter.generateSubResiduals();
        sysRhs_[_0] = splitter.takeDispVector();
        sysRhs_[_1] = splitter.takeRotVector();
        sysRhs_[_2] = splitter.takeSPresVector();

        sysRhs_[_0] *= lam1;
        sysRhs_[_1] *= lam2;
        sysRhs_[_2] *= lam2;

        const auto& prm = this->prm_;

        // !! OBS: Change back to needRebuild !!
        if (true) {
            OPM_TIMEBLOCK(flexibleSolverCreate);
            createSystemSolver(prm);
            sysInitialized_ = true;
        } else {
            OPM_TIMEBLOCK(flexibleSolverUpdate);
            sysPrecond_->update();
        }
    }

    void createSystemSolver(const PropertyTree& prm)
    {
        // Derive weights from the reservoir sub-block config (which uses CPR internally)
        std::function<SystemVectorT<Scalar>()> sysWeightCalc;

        const bool is_parallel = this->comm_->communicator().size() > 1;
        if (is_parallel) {
#if HAVE_MPI
            // !! OBS: Purge wellComm??? !!
            wellComm_ = std::make_unique<WellComm>();
            systemComm_ = std::make_unique<SystemComm>(*(this->comm_), *wellComm_);

            sysOpPar_ = std::make_unique<SystemParOpT<Scalar>>(sysMatrix_, *systemComm_);

            sysFlexSolverPar_ = std::make_unique<Dune::FlexibleSolver<SystemParOpT<Scalar>>>(
                *sysOpPar_, *systemComm_, prm, sysWeightCalc, pressureIndex);

            sysSolver_ = sysFlexSolverPar_.get();
            sysPrecond_ = &sysFlexSolverPar_->preconditioner();
#endif
        } else {
            sysOp_ = std::make_unique<SystemSeqOpT<Scalar>>(sysMatrix_);

            sysFlexSolverSeq_ = std::make_unique<Dune::FlexibleSolver<SystemSeqOpT<Scalar>>>(
                *sysOp_, prm, sysWeightCalc, pressureIndex);

            sysSolver_ = sysFlexSolverSeq_.get();
            sysPrecond_ = &sysFlexSolverSeq_->preconditioner();
        }
    }

    void postProcessSolution(Vector& x)
    {
        Scalar lam1 = 1.0e-5;
        Scalar lam2 = 1.0e5;
        for (std::size_t i = 0; i < x.size(); ++i) {
            // Displacement
            x[i][0] = sysX_[_0][i][0] * lam1;
            x[i][1] = sysX_[_0][i][1] * lam1;
            x[i][2] = sysX_[_0][i][2] * lam1;

            // Rotation
            x[i][3] = sysX_[_1][i][0] * lam2;
            x[i][4] = sysX_[_1][i][1] * lam2;
            x[i][5] = sysX_[_1][i][2] * lam2;

            // Solid pressure
            x[i][6] = sysX_[_2][i][0] * lam2;
        }
    }

    bool sysInitialized_ = false;

    SystemMatrixT<Scalar> sysMatrix_;
    SystemVectorT<Scalar> sysX_;
    SystemVectorT<Scalar> sysRhs_;
    std::vector<std::set<unsigned> > sparsityPattern_;

    // Pointers to take over ownership of submatrices
    std::unique_ptr<DispDispMatrixT<Scalar>> DDmat_{};
    std::unique_ptr<DispRotMatrixT<Scalar>> DRmat_{};
    std::unique_ptr<DispSPresMatrixT<Scalar>> DSmat_{};

    std::unique_ptr<RotDispMatrixT<Scalar>> RDmat_{};
    std::unique_ptr<RotRotMatrixT<Scalar>> RRmat_{};
    std::unique_ptr<RotSPresMatrixT<Scalar>> RSmat_{};

    std::unique_ptr<SPresDispMatrixT<Scalar>> SDmat_{};
    std::unique_ptr<SPresRotMatrixT<Scalar>> SRmat_{};
    std::unique_ptr<SPresSPresMatrixT<Scalar>> SSmat_{};

    // Serial solver components
    std::unique_ptr<SystemSeqOpT<Scalar>> sysOp_;
    std::unique_ptr<Dune::FlexibleSolver<SystemSeqOpT<Scalar>>> sysFlexSolverSeq_;

    // Parallel solver components
#if HAVE_MPI
    using WellComm = Dune::JacComm;
    std::unique_ptr<WellComm> wellComm_;
    std::unique_ptr<SystemComm> systemComm_;
    std::unique_ptr<SystemParOpT<Scalar>> sysOpPar_;
    std::unique_ptr<Dune::FlexibleSolver<SystemParOpT<Scalar>>> sysFlexSolverPar_;
#endif

    using SysSolverType = Dune::InverseOperator<SystemVectorT<Scalar>, SystemVectorT<Scalar>>;
    using SysPrecondType = Dune::PreconditionerWithUpdate<SystemVectorT<Scalar>, SystemVectorT<Scalar>>;
    SysSolverType* sysSolver_ = nullptr;
    SysPrecondType* sysPrecond_ = nullptr;
};

} // namespace Opm
