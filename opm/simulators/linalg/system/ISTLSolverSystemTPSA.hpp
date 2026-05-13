#pragma once

#include "SystemPreconditionerFactoryTPSA.hpp"
#include "SystemTypes.hpp"
#include "MatrixResidualSplitterHypreTPSA.hpp"

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

    using Splitter = MatrixResidualSplitterHypreTPSA<Scalar, Matrix, Vector>;

    constexpr static std::size_t pressureIndex = 0;
    constexpr static Scalar scale = 1e5;

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
        sysX_[_0].resize(numCells * 3);
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

        // Create solver or update preconditioner
        if (needRebuild) {
            OPM_TIMEBLOCK(flexibleSolverCreate);
            generateSubMatricesAndResiduals();
            const auto& prm = this->prm_;
            createSystemSolver(prm);
            sysInitialized_ = true;
        } else {
            OPM_TIMEBLOCK(flexibleSolverUpdate);
            updateSubMatricesAndResiduals();
            sysPrecond_->update();
        }
    }

    void generateSubMatricesAndResiduals()
    {
        // Split parent matrix and residual into sub-matrices and residuals, one per equation/PV:
        // 3 displacement, 3 rotation, and 1 solid pressure
        Splitter splitter(*this->matrix_, *this->rhs_, sparsityPattern_);
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

        DDmat_->istlMatrix() /= scale * scale;
        RRmat_->istlMatrix() *= scale * scale;
        SSmat_->istlMatrix() *= scale * scale;

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

        sysRhs_[_0] /= scale;
        sysRhs_[_1] *= scale;
        sysRhs_[_2] *= scale;
    }

    void updateSubMatricesAndResiduals()
    {
        Splitter splitter(*this->matrix_, *this->rhs_, sparsityPattern_);

        splitter.assignSubMatrices(*DDmat_,
                                   *DRmat_,
                                   *DSmat_,
                                   *RDmat_,
                                   *RRmat_,
                                   *RSmat_,
                                   *SDmat_,
                                   *SRmat_,
                                   *SSmat_);

        DDmat_->istlMatrix() /= scale * scale;
        RRmat_->istlMatrix() *= scale * scale;
        SSmat_->istlMatrix() *= scale * scale;

        splitter.assignSubResiduals(sysRhs_[_0], sysRhs_[_1], sysRhs_[_2]);

        sysRhs_[_0] /= scale;
        sysRhs_[_1] *= scale;
        sysRhs_[_2] *= scale;
    }

    void createSystemSolver(const PropertyTree& prm)
    {
        // Dummy weights
        std::function<SystemVectorT<Scalar>()> sysWeightCalc;

        if (this->comm_->communicator().size() > 1) {
#if HAVE_MPI
            systemComm_ = std::make_unique<SystemComm>(*this->comm_,
                                                       *this->comm_,
                                                       *this->comm_);

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
        for (std::size_t i = 0; i < x.size(); ++i) {
            // Displacement
            x[i][0] = sysX_[_0][3 * i][0] / scale;
            x[i][1] = sysX_[_0][3 * i + 1][0] / scale;
            x[i][2] = sysX_[_0][3 * i + 2][0] / scale;

            // Rotation
            x[i][3] = sysX_[_1][i][0] * scale;
            x[i][4] = sysX_[_1][i][1] * scale;
            x[i][5] = sysX_[_1][i][2] * scale;

            // Solid pressure
            x[i][6] = sysX_[_2][i][0] * scale;
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
