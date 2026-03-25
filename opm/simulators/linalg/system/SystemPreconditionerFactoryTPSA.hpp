#ifndef SYSTEM_PRECONDITIONER_FACTORY_TPSA_HPP
#define SYSTEM_PRECONDITIONER_FACTORY_TPSA_HPP

#include "MultiComm.hpp"
#include "SystemPreconditionerTPSA.hpp"
#include "SystemTypes.hpp"

#include <opm/simulators/linalg/PreconditionerFactory.hpp>

#include <dune/istl/operators.hh>
#include <dune/istl/paamg/pinfo.hh>

namespace Opm
{

template <class Operator, class Comm, typename>
struct StandardPreconditioners;

template<typename Scalar>
using SystemSeqOpT = Dune::MatrixAdapter<SystemMatrixT<Scalar>,
                                         SystemVectorT<Scalar>,
                                         SystemVectorT<Scalar> >;

#if HAVE_MPI
using SystemComm = Dune::MultiCommunicator<
    const Dune::OwnerOverlapCopyCommunication<int, int>&,
    const Dune::OwnerOverlapCopyCommunication<int, int>&,
    const Dune::OwnerOverlapCopyCommunication<int, int>&>;

template<typename Scalar>
using SystemParOpT = Dune::OverlappingSchwarzOperator<SystemMatrixT<Scalar>, SystemVectorT<Scalar>,
                                                      SystemVectorT<Scalar>, SystemComm>;
#endif

// Full specialisations of StandardPreconditioners for the coupled system
// operators.  Partial specialisations would be ambiguous with the generic
// serial factory (!is_gpu_operator_v guard), so full specialisations are
// used; detail:: helpers factor out the shared logic.

namespace detail {

template<typename Scalar>
void addSystemTPSASeq()
{
    using O = SystemSeqOpT<Scalar>;
    using F = PreconditionerFactory<O, Dune::Amg::SequentialInformation>;
    using V = SystemVectorT<Scalar>;
    using P = PropertyTree;

    F::addCreator(
        "system_tpsa",
        [](const O& op,
           const P& prm,
           [[maybe_unused]] const std::function<V()>& sysWeightCalc,
           [[maybe_unused]] std::size_t pressureIndex) {
            using PreCond = SystemPreconditionerTPSA<Scalar,
                                                     SeqDispDispOperatorT<Scalar>,
                                                     SeqRotRotOperatorT<Scalar>,
                                                     SeqSPresSPresOperatorT<Scalar> >;
            return std::make_shared<PreCond>(op.getmat(), prm);
        }
    );
}

#if HAVE_MPI
    template<typename Scalar>
    void addSystemTPSAParSeq()
    {
        using O = SystemSeqOpT<Scalar>;
        using F = PreconditionerFactory<O, Dune::Amg::SequentialInformation>;
        using V = SystemVectorT<Scalar>;
        using P = PropertyTree;

        F::addCreator(
            "system_tpsa",
            [](const O& op,
               const P& prm,
               [[maybe_unused]] const std::function<V()>& sysWeightCalc,
               [[maybe_unused]] std::size_t pressureIndex) {
                using PreCond = SystemPreconditionerTPSA<Scalar,
                                                         SeqDispDispOperatorT<Scalar>,
                                                         SeqRotRotOperatorT<Scalar>,
                                                         SeqSPresSPresOperatorT<Scalar> >;
                return std::make_shared<PreCond>(op.getmat(), prm);
            }
        );
    }

    template<typename Scalar>
    void addSystemTPSAPar()
    {
        using O = SystemParOpT<Scalar>;
        using F = PreconditionerFactory<O, SystemComm>;
        using V = SystemVectorT<Scalar>;
        using P = PropertyTree;

        F::addCreator(
            "system_tpsa",
            [](const O& op,
               const P& prm,
               [[maybe_unused]] const std::function<V()>& sysWeightCalc,
               [[maybe_unused]] std::size_t pressureIndex,
               const SystemComm& comm) {
                const auto& inComm = comm[Dune::Indices::_0];
                using PreCond = SystemPreconditionerTPSA<Scalar,
                                                         ParDispDispOperatorT<Scalar>,
                                                         ParRotRotOperatorT<Scalar>,
                                                         ParSPresSPresOperatorT<Scalar>,
                                                         ParComm>;
                return std::make_shared<PreCond>(op.getmat(), prm, inComm);
            }
        );
    }
#endif

} // namespace detail

template <>
struct StandardPreconditioners<SystemSeqOpT<double>, Dune::Amg::SequentialInformation, void> {
    static void add() { detail::addSystemTPSASeq<double>(); }
};

template <>
struct StandardPreconditioners<SystemSeqOpT<float>, Dune::Amg::SequentialInformation, void> {
    static void add() { detail::addSystemTPSASeq<float>(); }
};

#if HAVE_MPI
template <>
struct StandardPreconditioners<SystemParOpT<double>, Dune::Amg::SequentialInformation, void> {
    static void add() { detail::addSystemTPSAParSeq<double>(); }
};

template <>
struct StandardPreconditioners<SystemParOpT<float>, Dune::Amg::SequentialInformation, void> {
    static void add() { detail::addSystemTPSAParSeq<float>(); }
};

template <>
struct StandardPreconditioners<SystemParOpT<double>, SystemComm, void> {
    static void add() { detail::addSystemTPSAPar<double>(); }
};

template <>
struct StandardPreconditioners<SystemParOpT<float>, SystemComm, void> {
    static void add() { detail::addSystemTPSAPar<float>(); }
};
#endif

} // namespace Opm

#endif //SYSTEM_PRECONDITIONER_FACTORY_TPSA_HPP
