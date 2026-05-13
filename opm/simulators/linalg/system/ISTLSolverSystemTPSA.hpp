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
    static constexpr auto _3 = Dune::Indices::_3;
    static constexpr auto _4 = Dune::Indices::_4;

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
        sysX_[_3].resize(numCells);
        sysX_[_3] = 0.0;
        sysX_[_4].resize(numCells);
        sysX_[_4] = 0.0;

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
        DDmat00_ = splitter.takeDispDisp00Matrix();
        DDmat11_ = splitter.takeDispDisp11Matrix();
        DDmat22_ = splitter.takeDispDisp22Matrix();

        DRmat0_ = splitter.takeDispRot0Matrix();
        DRmat1_ = splitter.takeDispRot1Matrix();
        DRmat2_ = splitter.takeDispRot2Matrix();

        DSmat0_ = splitter.takeDispSPres0Matrix();
        DSmat1_ = splitter.takeDispSPres1Matrix();
        DSmat2_ = splitter.takeDispSPres2Matrix();

        RDmat0_ = splitter.takeRotDisp0Matrix();
        RDmat1_ = splitter.takeRotDisp1Matrix();
        RDmat2_ = splitter.takeRotDisp2Matrix();

        RRmat_ = splitter.takeRotRotMatrix();
        RSmat_ = splitter.takeRotSPresMatrix();

        SDmat0_ = splitter.takeSPresDisp0Matrix();
        SDmat1_ = splitter.takeSPresDisp1Matrix();
        SDmat2_ = splitter.takeSPresDisp2Matrix();

        SRmat_ = splitter.takeSPresRotMatrix();
        SSmat_ = splitter.takeSPresSPresMatrix();

        DDmat00_->istlMatrix() /= scale * scale;
        DDmat11_->istlMatrix() /= scale * scale;
        DDmat22_->istlMatrix() /= scale * scale;
        RRmat_->istlMatrix() *= scale * scale;
        SSmat_->istlMatrix() *= scale * scale;

        sysMatrix_.M11_00 = DDmat00_.get();
        sysMatrix_.M11_11 = DDmat11_.get();
        sysMatrix_.M11_22 = DDmat22_.get();

        sysMatrix_.M12_00 = DRmat0_.get();
        sysMatrix_.M12_10 = DRmat1_.get();
        sysMatrix_.M12_20 = DRmat2_.get();

        sysMatrix_.M13_00 = DSmat0_.get();
        sysMatrix_.M13_10 = DSmat1_.get();
        sysMatrix_.M13_20 = DSmat2_.get();

        sysMatrix_.M21_00 = RDmat0_.get();
        sysMatrix_.M21_01 = RDmat1_.get();
        sysMatrix_.M21_02 = RDmat2_.get();

        sysMatrix_.M22 = RRmat_.get();
        sysMatrix_.M23 = RSmat_.get();

        sysMatrix_.M31_00 = SDmat0_.get();
        sysMatrix_.M31_01 = SDmat1_.get();
        sysMatrix_.M31_02 = SDmat2_.get();

        sysMatrix_.M32 = SRmat_.get();
        sysMatrix_.M33 = SSmat_.get();

        // Set sub-residuals
        splitter.generateSubResiduals();
        sysRhs_[_0] = splitter.takeDisp0Vector();
        sysRhs_[_1] = splitter.takeDisp1Vector();
        sysRhs_[_2] = splitter.takeDisp2Vector();
        sysRhs_[_3] = splitter.takeRotVector();
        sysRhs_[_4] = splitter.takeSPresVector();

        sysRhs_[_0] /= scale;
        sysRhs_[_1] /= scale;
        sysRhs_[_2] /= scale;
        sysRhs_[_3] *= scale;
        sysRhs_[_4] *= scale;
    }

    void updateSubMatricesAndResiduals()
    {
        Splitter splitter(*this->matrix_, *this->rhs_, sparsityPattern_);

        splitter.assignSubMatrices(*DDmat00_,
                                   *DDmat11_,
                                   *DDmat22_,
                                   *DRmat0_,
                                   *DRmat1_,
                                   *DRmat2_,
                                   *DSmat0_,
                                   *DSmat1_,
                                   *DSmat2_,
                                   *RDmat0_,
                                   *RDmat1_,
                                   *RDmat2_,
                                   *RRmat_,
                                   *RSmat_,
                                   *SDmat0_,
                                   *SDmat1_,
                                   *SDmat2_,
                                   *SRmat_,
                                   *SSmat_);

        DDmat00_->istlMatrix() /= scale * scale;
        DDmat11_->istlMatrix() /= scale * scale;
        DDmat22_->istlMatrix() /= scale * scale;
        RRmat_->istlMatrix() *= scale * scale;
        SSmat_->istlMatrix() *= scale * scale;

        splitter.assignSubResiduals(sysRhs_[_0],
                                    sysRhs_[_1],
                                    sysRhs_[_2],
                                    sysRhs_[_3],
                                    sysRhs_[_4]);

        sysRhs_[_0] /= scale;
        sysRhs_[_1] /= scale;
        sysRhs_[_2] /= scale;
        sysRhs_[_3] *= scale;
        sysRhs_[_4] *= scale;
    }

    void createSystemSolver(const PropertyTree& prm)
    {
        // Dummy weights
        std::function<SystemVectorT<Scalar>()> sysWeightCalc;

        const bool is_parallel = this->comm_->communicator().size() > 1;
        if (is_parallel) {
#if HAVE_MPI
            systemComm_ = std::make_unique<SystemComm>(*this->comm_,
                                                       *this->comm_,
                                                       *this->comm_,
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
            x[i][0] = sysX_[_0][i][0] / scale;
            x[i][1] = sysX_[_1][i][0] / scale;
            x[i][2] = sysX_[_2][i][0] / scale;

            // Rotation
            x[i][3] = sysX_[_3][i][0] * scale;
            x[i][4] = sysX_[_3][i][1] * scale;
            x[i][5] = sysX_[_3][i][2] * scale;

            // Solid pressure
            x[i][6] = sysX_[_4][i][0] * scale;
        }
    }

    bool sysInitialized_ = false;

    SystemMatrixT<Scalar> sysMatrix_;
    SystemVectorT<Scalar> sysX_;
    SystemVectorT<Scalar> sysRhs_;
    std::vector<std::set<unsigned> > sparsityPattern_;

    // Pointers to take over ownership of submatrices
    std::unique_ptr<DispDispMatrix00T<Scalar> > DDmat00_{};
    std::unique_ptr<DispDispMatrix11T<Scalar> > DDmat11_{};
    std::unique_ptr<DispDispMatrix22T<Scalar> > DDmat22_{};
    std::unique_ptr<DispRotMatrix0T<Scalar> > DRmat0_{};
    std::unique_ptr<DispRotMatrix1T<Scalar> > DRmat1_{};
    std::unique_ptr<DispRotMatrix2T<Scalar> > DRmat2_{};
    std::unique_ptr<DispSPresMatrix0T<Scalar> > DSmat0_{};
    std::unique_ptr<DispSPresMatrix1T<Scalar> > DSmat1_{};
    std::unique_ptr<DispSPresMatrix2T<Scalar> > DSmat2_{};

    std::unique_ptr<RotDispMatrix0T<Scalar> > RDmat0_{};
    std::unique_ptr<RotDispMatrix1T<Scalar> > RDmat1_{};
    std::unique_ptr<RotDispMatrix2T<Scalar> > RDmat2_{};
    std::unique_ptr<RotRotMatrixT<Scalar> > RRmat_{};
    std::unique_ptr<RotSPresMatrixT<Scalar> > RSmat_{};

    std::unique_ptr<SPresDispMatrix0T<Scalar> > SDmat0_{};
    std::unique_ptr<SPresDispMatrix1T<Scalar> > SDmat1_{};
    std::unique_ptr<SPresDispMatrix2T<Scalar> > SDmat2_{};
    std::unique_ptr<SPresRotMatrixT<Scalar> > SRmat_{};
    std::unique_ptr<SPresSPresMatrixT<Scalar> > SSmat_{};

    // Serial solver components
    std::unique_ptr<SystemSeqOpT<Scalar> > sysOp_;
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
