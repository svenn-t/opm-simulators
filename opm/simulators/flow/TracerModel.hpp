// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/**
 * \file
 *
 * \copydoc Opm::TracerModel
 */
#ifndef OPM_TRACER_MODEL_HPP
#define OPM_TRACER_MODEL_HPP

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/TimingMacros.hpp>

#include <opm/input/eclipse/EclipseState/Aquifer/AquiferConfig.hpp>
#include <opm/input/eclipse/EclipseState/Grid/FaceDir.hpp>
#include <opm/input/eclipse/Schedule/BCState.hpp>
#include <opm/input/eclipse/Schedule/Schedule.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>
#include <opm/input/eclipse/Schedule/Well/WellConnections.hpp>

#include <opm/material/fluidstates/BlackOilFluidState.hpp>
#include <opm/material/fluidsystems/BlackOilFunctions.hpp>

#include <opm/models/utils/parametersystem.hpp>
#include <opm/models/utils/propertysystem.hh>

#include <opm/simulators/flow/FlowProblemParameters.hpp>
#include <opm/simulators/flow/GenericTracerModel.hpp>
#include <opm/simulators/flow/TransportLinearSystem.hpp>
#include <opm/simulators/utils/DeferredLoggingErrorHelpers.hpp>
#include <opm/simulators/utils/gatherDeferredLogger.hpp>
#include <opm/simulators/utils/VectorVectorDataHandle.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace Opm::Properties {

template<class TypeTag, class MyTypeTag>
struct EnableTracerModel {
    using type = UndefinedProperty;
};

} // namespace Opm::Properties

namespace Opm {

/*!
 * \ingroup BlackOilSimulator
 *
 * \brief A class which handles tracers as specified in by ECL
 */
template <class TypeTag>
class TracerModel : public GenericTracerModel<GetPropType<TypeTag, Properties::Grid>,
                                              GetPropType<TypeTag, Properties::GridView>,
                                              GetPropType<TypeTag, Properties::DofMapper>,
                                              GetPropType<TypeTag, Properties::Stencil>,
                                              GetPropType<TypeTag, Properties::FluidSystem>,
                                              GetPropType<TypeTag, Properties::Scalar>>
{
    using BaseType = GenericTracerModel<GetPropType<TypeTag, Properties::Grid>,
                                        GetPropType<TypeTag, Properties::GridView>,
                                        GetPropType<TypeTag, Properties::DofMapper>,
                                        GetPropType<TypeTag, Properties::Stencil>,
                                        GetPropType<TypeTag, Properties::FluidSystem>,
                                        GetPropType<TypeTag, Properties::Scalar>>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using Grid = GetPropType<TypeTag, Properties::Grid>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Stencil = GetPropType<TypeTag, Properties::Stencil>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using IntensiveQuantities = GetPropType<TypeTag, Properties::IntensiveQuantities>;
    using ExtensiveQuantities = GetPropType<TypeTag, Properties::ExtensiveQuantities>;

    using TracerMatrix = typename BaseType::TracerMatrix;
    using TracerVector = typename BaseType::TracerVector;
    using TracerVectorSingle = typename BaseType::TracerVectorSingle;

    //! \brief Static data of an interior face, see phaseVolumeFlux_().
    struct FaceInfo
    {
        Scalar dZg{}; //!< Depth difference (interior - exterior) times gravity
        Scalar Vin{}; //!< Bulk volume of the interior cell
        Scalar Vex{}; //!< Bulk volume of the exterior cell
        FaceDir::DirEnum faceDir{FaceDir::DirEnum::Unknown};
    };

    using LinearSystem = TransportLinearSystem<Scalar, 2, FaceInfo>;
    using NeighborInfo = typename LinearSystem::NeighborInfo;
    using BoundaryFace = typename LinearSystem::BoundaryFace;
    static_assert(std::is_same_v<typename LinearSystem::Matrix, TracerMatrix>);

    //! \brief Surface volume rate of a tracer carrier across a face, out of the cell.
    struct FaceFlux
    {
        Scalar rate{0.0};
        bool upIsInterior{false}; //!< Whether the cell is upstream
    };

    enum { numEq = getPropValue<TypeTag, Properties::NumEq>() };
    enum { numPhases = FluidSystem::numPhases };
    enum { waterPhaseIdx = FluidSystem::waterPhaseIdx };
    enum { oilPhaseIdx = FluidSystem::oilPhaseIdx };
    enum { gasPhaseIdx = FluidSystem::gasPhaseIdx };
    enum { enableDiffusion = getPropValue<TypeTag, Properties::EnableDiffusion>() };
    enum { enableDispersion = getPropValue<TypeTag, Properties::EnableDispersion>() };

public:
    explicit TracerModel(Simulator& simulator)
        : BaseType(simulator.vanguard().gridView(),
                   simulator.vanguard().eclState(),
                   simulator.vanguard().cartesianIndexMapper(),
                   simulator.model().dofMapper(),
                   simulator.vanguard().cellCentroids())
        , simulator_(simulator)
        , dispersionActive_(simulator.vanguard().eclState().getSimulationConfig().rock_config().dispersion())
    { }


    /*
      The initialization of the tracer model is a three step process:

      1. The init() method is called. This will allocate buffers and initialize
         some phase index stuff. If this is a normal run the initial tracer
         concentrations will be assigned from the TBLK or TVDPF keywords.

      2. [Restart only:] The tracer concentration are read from the restart
         file and the concentrations are applied with repeated calls to the
         setTracerConcentration() method. This is currently done in the
         eclwriter::beginRestart() method.

      3. Internally the tracer model manages the concentrations in "batches" for
         the oil, water and gas tracers respectively. The batches should be
         initialized with the initial concentration, that must be performed
         after the concentration values have been assigned. This is done in
         method prepareTracerBatches() called from eclproblem::finishInit().
    */
    void init(bool rst)
    {
        this->doInit(rst, simulator_.model().numGridDof(),
                     gasPhaseIdx, oilPhaseIdx, waterPhaseIdx);

        const std::string solver = Parameters::Get<Parameters::TracerSolver>();
        if (solver != "istl" && solver != "reorder") {
            throw std::invalid_argument("Unknown tracer solver '" + solver +
                                        "', use 'istl' or 'reorder'");
        }
        this->useReorderSolver_ = solver == "reorder";

        // The reorder solver cannot be used for diffusion/dispersion fluxes
        if (this->useReorderSolver_) {
            bool hasDispersion = false;
            if constexpr (enableDispersion) {
                hasDispersion = dispersionActive_ && !this->tracerPhaseIdx_.empty();
            }
            bool hasDiffusion = false;
            if constexpr (enableDiffusion) {
                for (std::size_t tracerIdx = 0; tracerIdx < this->tracerPhaseIdx_.size();
                     ++tracerIdx) {
                    const auto& tracerConfig = this->eclState_.tracer()[tracerIdx];
                    hasDiffusion = hasDiffusion || tracerConfig.free_diffusion_coefficient != 0.0 ||
                                   (hasSolutionPart_(this->tracerPhaseIdx_[tracerIdx]) &&
                                    tracerConfig.solution_diffusion_coefficient != 0.0);
                }
            }
            if (hasDiffusion || hasDispersion) {
                const std::string unsupported = hasDiffusion && hasDispersion
                    ? "diffusion (TRDIF) and dispersion (DISPERC)"
                    : (hasDiffusion ? "diffusion (TRDIF)" : "dispersion (DISPERC)");
                throw std::invalid_argument("The reorder tracer solver does not support " +
                                            unsupported + ", use --tracer-solver=istl");
            }
        }

        reorderRetrySteps_ = Parameters::Get<Parameters::TracerReorderRetrySteps>();
        if (reorderRetrySteps_ < 0) {
            throw std::invalid_argument("The number of tracer reorder retry steps must not be "
                                        "negative, got " + std::to_string(reorderRetrySteps_));
        }
    }

    void prepareTracerBatches()
    {
        DeferredLogger local_deferredLogger;

        // Tracers of the same phase and with the same diffusion coefficients (TRDIFxxx) are
        // solved together in a batch, with a common system matrix and separate right-hand
        // sides. Tracers with different diffusion coefficients get separate batches.
        for (std::size_t tracerIdx = 0; tracerIdx < this->tracerPhaseIdx_.size(); ++tracerIdx) {
            const int phaseIdx = this->tracerPhaseIdx_[tracerIdx];
            if (phaseIdx == FluidSystem::waterPhaseIdx) {
                if (! FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)){
                    throw std::runtime_error("Water tracer specified for non-water fluid system: " +
                                             this->name(tracerIdx));
                }
            }
            else if (phaseIdx == FluidSystem::oilPhaseIdx) {
                if (! FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)){
                    throw std::runtime_error("Oil tracer specified for non-oil fluid system: " +
                                             this->name(tracerIdx));
                }
            }
            else if (phaseIdx == FluidSystem::gasPhaseIdx) {
                if (! FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)){
                    throw std::runtime_error("Gas tracer specified for non-gas fluid system: " +
                                             this->name(tracerIdx));
                }
            }

            // Molecular diffusion coefficients of the free tracer and of the tracer in solution
            // (TRDIFxxx). Tracers without solution part (e.g. water tracers) only have the free one.
            const auto& tracerConfig = this->eclState_.tracer()[tracerIdx];
            const Scalar freeD = tracerConfig.free_diffusion_coefficient;
            const Scalar solutionD = hasSolutionPart_(phaseIdx) ? tracerConfig.solution_diffusion_coefficient
                                                                : freeD;
            if (freeD != 0.0 || solutionD != 0.0) {
                // The diffusion of the tracers needs the diffusivity of the faces, which is only
                // available if diffusion is compiled in and activated with DIFFUSE.
                const std::string diffKeyword = "TRDIF" + this->name(tracerIdx);
                if constexpr (!enableDiffusion) {
                    throw std::runtime_error(diffKeyword + " is given for tracer " + this->name(tracerIdx) +
                                             ", but this simulator was built without diffusion");
                }
                if (!this->eclState_.getSimulationConfig().isDiffusive()) {
                    throw std::runtime_error(diffKeyword + " is given for tracer " + this->name(tracerIdx) +
                                             ", which requires the DIFFUSE keyword");
                }
            }
            auto batch = std::ranges::find_if(this->tbatch,
                                              [phaseIdx, freeD, solutionD](const auto& tr)
                                              {
                                                  return tr.phaseIdx_ == phaseIdx &&
                                                         tr.freeDiffusionCoefficient_ == freeD &&
                                                         tr.solutionDiffusionCoefficient_ == solutionD;
                                              });
            if (batch == this->tbatch.end()) {
                this->tbatch.emplace_back(phaseIdx);
                batch = std::prev(this->tbatch.end());
                batch->freeDiffusionCoefficient_ = freeD;
                batch->solutionDiffusionCoefficient_ = solutionD;
            }
            batch->addTracer(tracerIdx, this->tracerConcentration_[tracerIdx]);

            // resize free and solution volume storages
            vol1_[0][phaseIdx].resize(this->freeTracerConcentration_[tracerIdx].size());
            vol1_[1][phaseIdx].resize(this->freeTracerConcentration_[tracerIdx].size());
        }

        if (this->numTracers() != 0) {
            initLinearSystem_();
        }

        this->buildAquiferTracerConnections_(local_deferredLogger);
        this->checkBoundaryTracers_(local_deferredLogger);

        const auto& comm = simulator_.vanguard().grid().comm();
        auto global_logger = gatherDeferredLogger(local_deferredLogger, comm);
        if (comm.rank() == 0) {
            global_logger.logMessages();
        }
    }

    void beginTimeStep()
    {
        if (this->numTracers() == 0) {
            return;
        }

        OPM_TIMEBLOCK(tracerUpdateCache);
        updateStorageCache();
    }

    /*!
     * \brief Informs the tracer model that a time step has just been finished.
     */
    void endTimeStep()
    {
        if (this->numTracers() == 0) {
            return;
        }

        OPM_TIMEBLOCK(tracerAdvance);
        advanceTracerFields();
    }

    /*!
     * \brief This method writes the complete state of all tracer
     *        to the hard disk.
     */
    template <class Restarter>
    void serialize(Restarter&)
    { /* not implemented */ }

    /*!
     * \brief This method restores the complete state of the tracer
     *        from disk.
     *
     * It is the inverse of the serialize() method.
     */
    template <class Restarter>
    void deserialize(Restarter&)
    { /* not implemented */ }

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(static_cast<BaseType&>(*this));
        serializer(tbatch);
    }

protected:
    using TracerTypeIdx = typename BaseType::TracerTypeIdx;
    using BaseType::Free;
    using BaseType::Solution;

    // compute volume associated with free/solution concentration
    template<TracerTypeIdx Index>
    Scalar computeVolume_(const int tracerPhaseIdx,
                          const unsigned globalDofIdx,
                          const unsigned timeIdx) const
    {
        const auto& intQuants = simulator_.model().intensiveQuantities(globalDofIdx, timeIdx);
        const auto& fs = intQuants.fluidState();
        constexpr Scalar min_volume = 1e-10;

        if constexpr (Index == Free) {
            return std::max(decay<Scalar>(fs.saturation(tracerPhaseIdx)) *
                            decay<Scalar>(fs.invB(tracerPhaseIdx)) *
                            decay<Scalar>(intQuants.porosity()),
                            min_volume);
        } else {
            // vaporized oil
            if (tracerPhaseIdx == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                return std::max(decay<Scalar>(fs.saturation(FluidSystem::gasPhaseIdx)) *
                                decay<Scalar>(fs.invB(FluidSystem::gasPhaseIdx)) *
                                decay<Scalar>(fs.Rv()) *
                                decay<Scalar>(intQuants.porosity()),
                                min_volume);
            }

            // dissolved gas
            else if (tracerPhaseIdx == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                return std::max(decay<Scalar>(fs.saturation(FluidSystem::oilPhaseIdx)) *
                                decay<Scalar>(fs.invB(FluidSystem::oilPhaseIdx)) *
                                decay<Scalar>(fs.Rs()) *
                                decay<Scalar>(intQuants.porosity()),
                                min_volume);
            }

            return min_volume;
        }
    }

    /*!
     * \brief Set up the neighbor table and the matrix of the tracer equations.
     *
     * Only static data is stored for the faces. The transmissibilities, threshold
     * pressures, diffusivities and dispersivities are taken from the problem during the
     * assembly since they may change during the simulation.
     */
    void initLinearSystem_()
    {
        const auto& problem = simulator_.problem();
        const auto& model = simulator_.model();
        const Scalar gravity = problem.gravity()[GridView::dimensionworld - 1];
        system_.template init<Stencil>(this->gridView_, this->dofMapper_, model.numGridDof(),
            [&problem, &model, gravity](const unsigned globI, const unsigned globJ, const auto& scvf)
            {
                const auto dirId = scvf.dirId();
                return FaceInfo{(problem.dofCenterDepth(globI) - problem.dofCenterDepth(globJ)) * gravity,
                                model.dofTotalVolume(globI),
                                model.dofTotalVolume(globJ),
                                dirId < 0 ? FaceDir::DirEnum::Unknown
                                          : FaceDir::FromIntersectionIndex(dirId)};
            });
    }

    /*!
     * \brief Reservoir volume rate of a phase from cell I to its neighbor across a face.
     *
     * This is the two-point flux of the flow equations (see NewTranFluxModule),
     * multiplied by the face area.
     *
     * \return The rate and whether cell I is upstream.
     */
    std::pair<Scalar, bool>
    phaseVolumeFlux_(const unsigned phaseIdx,
                     const unsigned I,
                     const NeighborInfo& nbInfo) const
    {
        const auto& problem = simulator_.problem();
        const unsigned J = nbInfo.neighbor;
        const auto& face = nbInfo.res_nbinfo;
        const IntensiveQuantities& intQuantsIn = simulator_.model().intensiveQuantities(I, /*timeIdx=*/0);
        const IntensiveQuantities& intQuantsEx = simulator_.model().intensiveQuantities(J, /*timeIdx=*/0);

        constexpr short interiorDofIdx = 0;
        constexpr short exteriorDofIdx = 1;
        short upIdx = interiorDofIdx;
        short dnIdx = exteriorDofIdx;
        Evaluation pressureDifference;
        ExtensiveQuantities::calculatePhasePressureDiff_(upIdx,
                                                         dnIdx,
                                                         pressureDifference,
                                                         intQuantsIn,
                                                         intQuantsEx,
                                                         phaseIdx,
                                                         interiorDofIdx,
                                                         exteriorDofIdx,
                                                         face.Vin,
                                                         face.Vex,
                                                         I,
                                                         J,
                                                         face.dZg,
                                                         problem.thresholdPressure(I, J),
                                                         problem.thresholdPressure(J, I),
                                                         problem.moduleParams());

        const bool inIsUp = upIdx == interiorDofIdx;
        const IntensiveQuantities& up = inIsUp ? intQuantsIn : intQuantsEx;
        // Arithmetic average, as in the flow equations
        const Scalar transMult = (decay<Scalar>(intQuantsIn.rockCompTransMultiplier()) +
                                  decay<Scalar>(intQuantsEx.rockCompTransMultiplier())) / 2;
        const Scalar trans = problem.transmissibility(I, J);
        const Scalar rate = -decay<Scalar>(pressureDifference) *
                            decay<Scalar>(up.mobility(phaseIdx, face.faceDir)) *
                            transMult * trans;
        return {rate, inIsUp};
    }

    //! \brief Surface volume rate of the free/solution tracer carrier out of cell I across a face.
    template<TracerTypeIdx Index>
    FaceFlux computeFlux_(const int tracerPhaseIdx,
                          const unsigned I,
                          const NeighborInfo& nbInfo) const
    {
        if constexpr (Index == Solution) {
            if (!hasSolutionPart_(tracerPhaseIdx)) {
                return {};
            }
        }

        const int phaseIdx = carrierPhaseIdx_<Index>(tracerPhaseIdx);
        const auto [volumeRate, inIsUp] = phaseVolumeFlux_(phaseIdx, I, nbInfo);
        const unsigned upIdx = inIsUp ? I : nbInfo.neighbor;
        const auto& fs = simulator_.model().intensiveQuantities(upIdx, /*timeIdx=*/0).fluidState();
        return {volumeRate * carrierSurfaceVolume_<Index>(tracerPhaseIdx, fs, /*saturationWeighted=*/false),
                inIsUp};
    }

    /**
     * @brief Phase that carries the tracer of the given type: the phase of the tracer for the free
     *        tracer, the gas phase for the solution tracer of an oil tracer (vaporized oil) and the
     *        oil phase for the solution tracer of a gas tracer (dissolved gas).
     */
    template<TracerTypeIdx Index>
    static int carrierPhaseIdx_(const int tracerPhaseIdx)
    {
        if constexpr (Index == Free) {
            return tracerPhaseIdx;
        }
        else {
            return tracerPhaseIdx == FluidSystem::oilPhaseIdx ? FluidSystem::gasPhaseIdx
                                                             : FluidSystem::oilPhaseIdx;
        }
    }

    /**
     * @brief Whether the tracers of the phase have a solution part (vaporized oil, dissolved gas)
     */
    static bool hasSolutionPart_(const int tracerPhaseIdx)
    {
        return (tracerPhaseIdx == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) ||
               (tracerPhaseIdx == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas());
    }

    /**
     * @brief Surface volume of the tracer carrier per unit pore volume of a cell: the inverse formation
     *        volume factor of the carrier phase, times Rv or Rs for the solution tracer (the dissolved
     *        component per surface volume of the carrier) and, optionally, the saturation of the carrier.
     */
    template<TracerTypeIdx Index, class FluidState>
    static Scalar carrierSurfaceVolume_(const int tracerPhaseIdx,
                                        const FluidState& fs,
                                        const bool saturationWeighted)
    {
        const int phaseIdx = carrierPhaseIdx_<Index>(tracerPhaseIdx);
        Scalar w = decay<Scalar>(fs.invB(phaseIdx));
        if constexpr (Index == Solution) {
            w *= tracerPhaseIdx == FluidSystem::oilPhaseIdx ? decay<Scalar>(fs.Rv())
                                                            : decay<Scalar>(fs.Rs());
        }
        if (saturationWeighted) {
            w *= decay<Scalar>(fs.saturation(phaseIdx));
        }
        return w;
    }

    /**
     * @brief Whether the carrier of the tracer is present in a cell, i.e. there is carrier phase (and, for
     *        the solution tracer, dissolved or vaporized component) to hold tracer. The Fickian flux across
     *        a face is zero unless this is the case in both cells: averaging over the two cells would
     *        otherwise couple the tracer into a cell where its concentration has no meaning.
     */
    template<TracerTypeIdx Index, class FluidState>
    static bool carrierPresent_(const int tracerPhaseIdx, const FluidState& fs)
    {
        return carrierSurfaceVolume_<Index>(tracerPhaseIdx, fs, /*saturationWeighted=*/true) > 0.0;
    }

    /**
     * @brief Diffusive coefficient of a face for the Fickian flux
     *        J = -D_mol * diffusivity * w * (c_J - c_I), with the face diffusivity of the
     *        problem (which includes the face area). The geometric (porosity weighted)
     *        face diffusivity multiplies the molecular diffusion coefficient D_mol of the tracer
     *        (TRDIFxxx), and w is the face average of the saturation times the inverse formation
     *        volume factor of the carrier phase of the tracer (see carrierSurfaceVolume_()). The
     *        coefficient is zero unless the carrier is present in both cells (see carrierPresent_()).
     */
    template<TracerTypeIdx Index>
    Scalar computeDiffusivity_(const int tracerPhaseIdx,
                               const unsigned I,
                               const unsigned J) const
    {
        Scalar v = 0.0;
        if constexpr (enableDiffusion) {
            const auto& inFs = simulator_.model().intensiveQuantities(I, /*timeIdx=*/0).fluidState();
            const auto& outFs = simulator_.model().intensiveQuantities(J, /*timeIdx=*/0).fluidState();
            if (carrierPresent_<Index>(tracerPhaseIdx, inFs) && carrierPresent_<Index>(tracerPhaseIdx, outFs)) {
                v = simulator_.problem().diffusivity(I, J) *
                    0.5 * (carrierSurfaceVolume_<Index>(tracerPhaseIdx, inFs, /*saturationWeighted=*/true) +
                           carrierSurfaceVolume_<Index>(tracerPhaseIdx, outFs, /*saturationWeighted=*/true));
            }
        }

        return v;
    }

    /**
     * @brief Dispersive coefficient of a face for the Fickian flux
     *        J = -(w * |u| * disp) * (c_J - c_I), where disp is the geometric dispersivity
     *        (DISPERC, as in the blackoil dispersion module), |u| the average max-norm filter
     *        velocity of the carrier phase in the two cells, and w the face average of the
     *        inverse formation volume factor of the carrier phase (see carrierSurfaceVolume_()).
     *        The saturation is already in the filter velocity. Zero unless DISPERC is in the deck
     *        and the carrier is present in both cells (see carrierPresent_()).
     */
    template<TracerTypeIdx Index>
    Scalar computeDispersivity_(const int tracerPhaseIdx,
                                const unsigned I,
                                const unsigned J) const
    {
        Scalar flux = 0.0;
        if constexpr (enableDispersion) {
            if (this->dispersionActive_) {
                const int phaseIdx = carrierPhaseIdx_<Index>(tracerPhaseIdx);
                const auto& inIq = simulator_.model().intensiveQuantities(I, /*timeIdx=*/0);
                const auto& outIq = simulator_.model().intensiveQuantities(J, /*timeIdx=*/0);

                if (carrierPresent_<Index>(tracerPhaseIdx, inIq.fluidState()) &&
                    carrierPresent_<Index>(tracerPhaseIdx, outIq.fluidState())) {
                    const Scalar normVelocityAvg = 0.5 * (inIq.normVelocityCell(phaseIdx) +
                                                          outIq.normVelocityCell(phaseIdx));
                    const Scalar weight =
                        0.5 * (carrierSurfaceVolume_<Index>(tracerPhaseIdx, inIq.fluidState(), /*saturationWeighted=*/false) +
                               carrierSurfaceVolume_<Index>(tracerPhaseIdx, outIq.fluidState(), /*saturationWeighted=*/false));
                    flux = weight * normVelocityAvg * simulator_.problem().dispersivity(I, J);
                }
            }
        }

        return flux;
    }

    template<TracerTypeIdx Index, class TrRe>
    Scalar storage1_(const TrRe& tr,
                     const unsigned tIdx,
                     const unsigned I,
                     const unsigned I1,
                     const bool cache)
    {
        if (cache) {
            return tr.storageOfTimeIndex1_[tIdx][I][Index];
        } else {
            return computeVolume_<Index>(tr.phaseIdx_, I1, 1) *
                   tr.concentration_[tIdx][I1][Index];
        }
    }

    template<class TrRe>
    void assembleTracerEquationVolume(TrRe& tr,
                                      const Scalar scvVolume,
                                      const Scalar dt,
                                      const unsigned I,
                                      const bool enableStorageCache)
    {
        const Scalar fVol = computeVolume_<Free>(tr.phaseIdx_, I, 0);
        const Scalar sVol = computeVolume_<Solution>(tr.phaseIdx_, I, 0);
        tr.dVol_[Solution][I] += sVol * scvVolume - vol1_[1][tr.phaseIdx_][I];
        tr.dVol_[Free][I] += fVol * scvVolume - vol1_[0][tr.phaseIdx_][I];
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            // Free part
            const Scalar fStorageOfTimeIndex0 = fVol * tr.concentration_[tIdx][I][Free];
            const Scalar fLocalStorage = (fStorageOfTimeIndex0 - storage1_<Free>(tr, tIdx, I, I,
                                                                                 enableStorageCache)) * scvVolume / dt;
            tr.residual_[tIdx][I][Free] += fLocalStorage; // residual + flux

            // Solution part
            const Scalar sStorageOfTimeIndex0 = sVol * tr.concentration_[tIdx][I][Solution];
            const Scalar sLocalStorage = (sStorageOfTimeIndex0 - storage1_<Solution>(tr, tIdx, I, I,
                                                                                     enableStorageCache)) * scvVolume / dt;
            tr.residual_[tIdx][I][Solution] += sLocalStorage; // residual + flux
        }

        // Derivative matrix
        auto& diag = system_.diagonal(I);
        diag[Free][Free] += fVol * scvVolume / dt;
        diag[Solution][Solution] += sVol * scvVolume / dt;
    }

    /*!
     * \brief Add the advective flux of a tracer carrier out of cell I across a face.
     *
     * The tracer leaves with the concentration of cell I if cell I is upstream. Otherwise
     * it enters with the concentration given by \c upstreamConcentration(tIdx), which is
     * the concentration of the neighbor for interior faces. Only the dependency on the
     * concentration of cell I is added to the matrix.
     */
    template<TracerTypeIdx Index, class TrRe, class UpstreamConcentration>
    void addFaceFlux_(TrRe& tr,
                      const unsigned I,
                      const FaceFlux& flux,
                      const Scalar dt,
                      UpstreamConcentration&& upstreamConcentration)
    {
        tr.dVol_[Index][I] += flux.rate * dt;
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            const Scalar concentration = flux.upIsInterior ? tr.concentration_[tIdx][I][Index]
                                                           : upstreamConcentration(tIdx);
            tr.residual_[tIdx][I][Index] += flux.rate * concentration; // residual + flux
        }
        if (flux.upIsInterior) {
            system_.diagonal(I)[Index][Index] += flux.rate;
        }
    }

    template<class TrRe>
    void assembleTracerEquationFlux(TrRe& tr,
                                    const unsigned I,
                                    const NeighborInfo& nbInfo,
                                    const Scalar dt)
    {
        const unsigned J = nbInfo.neighbor;
        const FaceFlux fFlux = computeFlux_<Free>(tr.phaseIdx_, I, nbInfo);
        const FaceFlux sFlux = computeFlux_<Solution>(tr.phaseIdx_, I, nbInfo);

        // Fickian coefficients of the free and the solution tracer: diffusion (the geometric diffusivity
        // scaled by the molecular diffusion coefficient of the free and the solution tracer of the
        // batch, TRDIFxxx) plus dispersion (DISPERC). Only evaluated if diffusion or dispersion is
        // active, as this requires extra quantities for every face.
        const bool hasSolutionPart = hasSolutionPart_(tr.phaseIdx_);
        const bool hasDispersion = enableDispersion && this->dispersionActive_;
        const bool hasDiffusionFree = enableDiffusion && tr.freeDiffusionCoefficient_ != 0.0;
        const bool hasDiffusionSolution = enableDiffusion && hasSolutionPart &&
                                          tr.solutionDiffusionCoefficient_ != 0.0;
        const bool hasFickianFree = hasDiffusionFree || hasDispersion;
        const bool hasFickianSolution = hasDiffusionSolution || (hasDispersion && hasSolutionPart);
        Scalar fickianFree = 0.0;
        Scalar fickianSolution = 0.0;
        if (hasDiffusionFree) {
            fickianFree += computeDiffusivity_<Free>(tr.phaseIdx_, I, J) * tr.freeDiffusionCoefficient_;
        }
        if (hasDiffusionSolution) {
            fickianSolution += computeDiffusivity_<Solution>(tr.phaseIdx_, I, J) *
                               tr.solutionDiffusionCoefficient_;
        }
        if (hasDispersion) {
            fickianFree += computeDispersivity_<Free>(tr.phaseIdx_, I, J);
            if (hasSolutionPart) {
                fickianSolution += computeDispersivity_<Solution>(tr.phaseIdx_, I, J);
            }
        }

        // Advective fluxes, carried by the concentration of the upstream cell
        addFaceFlux_<Free>(tr, I, fFlux, dt,
                           [&tr, J](const int tIdx) { return tr.concentration_[tIdx][J][Free]; });
        addFaceFlux_<Solution>(tr, I, sFlux, dt,
                               [&tr, J](const int tIdx) { return tr.concentration_[tIdx][J][Solution]; });

        // Fickian flux J = -(D + D_disp) * grad(c), implicit in the tracer concentration
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            if (hasFickianFree) {
                const Scalar gradientFree = tr.concentration_[tIdx][J][Free] - tr.concentration_[tIdx][I][Free];
                tr.residual_[tIdx][I][Free] -= fickianFree * gradientFree;
            }
            if (hasFickianSolution) {
                const Scalar gradientSolution = tr.concentration_[tIdx][J][Solution] - tr.concentration_[tIdx][I][Solution];
                tr.residual_[tIdx][I][Solution] -= fickianSolution * gradientSolution;
            }
        }

        // Derivative matrix, as in the TPFA linearizer of the main simulator: this cell (I) writes
        // d(residual_I)/d(c_I) into the diagonal (see addFaceFlux_()) and
        // d(residual_J)/d(c_I) = -d(residual_I)/d(c_I) into the block (J, I), i.e. the row of the
        // neighbor J and the column of I. The block (J, I) is only written when assembling cell I,
        // which makes the assembly thread safe. Rows of cells that are not interior are Dirichlet
        // rows, so the matrix is decoupled between the ranks.
        auto& offDiag = *nbInfo.matBlockAddress;
        if (fFlux.upIsInterior) {
            offDiag[Free][Free] -= fFlux.rate;
        }
        if (sFlux.upIsInterior) {
            offDiag[Solution][Solution] -= sFlux.rate;
        }

        // Diffusion and dispersion: d(residual_I)/d(c_I) = +D and d(residual_J)/d(c_I) = -D,
        // for every face irrespective of the flow direction.
        if (hasFickianFree) {
            system_.diagonal(I)[Free][Free] += fickianFree;
            offDiag[Free][Free] -= fickianFree;
        }
        if (hasFickianSolution) {
            system_.diagonal(I)[Solution][Solution] += fickianSolution;
            offDiag[Solution][Solution] -= fickianSolution;
        }
    }

    template<class TrRe, class Well>
    void assembleTracerEquationWell(TrRe& tr,
                                    const Well& well)
    {
        if (tr.numTracer() == 0) {
            return;
        }

        const auto& eclWell = well.wellEcl();

        // Init. well output to zero
        auto& tracerRate = this->wellTracerRate_[eclWell.seqIndex()];
        auto& solTracerRate = this->wellSolTracerRate_[eclWell.seqIndex()];
        auto& freeTracerRate = this->wellFreeTracerRate_[eclWell.seqIndex()];
        auto* mswTracerRate = eclWell.isMultiSegment()
            ? &this->mSwTracerRate_[eclWell.seqIndex()]
            : nullptr;
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            tracerRate[tr.idx_[tIdx]] = {this->name(tr.idx_[tIdx]), 0.0};
            freeTracerRate[tr.idx_[tIdx]] = {this->wellfname(tr.idx_[tIdx]), 0.0};
            solTracerRate[tr.idx_[tIdx]] = {this->wellsname(tr.idx_[tIdx]), 0.0};
            if (eclWell.isMultiSegment()) {
                auto& wtr = mswTracerRate->at(tr.idx_[tIdx]) = {this->name(tr.idx_[tIdx])};;
                wtr.rate.reserve(eclWell.getConnections().size());
                for (std::size_t i = 0; i < eclWell.getConnections().size(); ++i) {
                    wtr.rate.emplace(eclWell.getConnections().get(i).segment(), 0.0);
                }
            }
        }

        std::vector<Scalar> wtracer(tr.numTracer());
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            wtracer[tIdx] = this->currentConcentration_(eclWell, this->name(tr.idx_[tIdx]),
                                                        simulator_.problem().wellModel().summaryState());
        }

        const Scalar dt = simulator_.timeStepSize();
        const auto& ws = simulator_.problem().wellModel().wellState().well(well.name());
        const auto well_eff = well.wellEfficiencyFactor(); // Needed to convert ws.perf_data.phase_mixing_rates to effective rates
        for (std::size_t i = 0; i < ws.perf_data.size(); ++i) {
            const auto I = ws.perf_data.cell_index[i];
            const Scalar rate = well.volumetricSurfaceRateForConnection(I, tr.phaseIdx_); // Includes (accumulated) well efficiency factor
            Scalar rate_s;

            if (tr.phaseIdx_ == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                rate_s = ws.perf_data.phase_mixing_rates[i][ws.vaporized_oil] * well_eff;
            }
            else if (tr.phaseIdx_ == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                rate_s = ws.perf_data.phase_mixing_rates[i][ws.dissolved_gas] * well_eff;
            }
            else {
                rate_s = 0.0;
            }

            const Scalar rate_f = rate - rate_s;
            if (rate_f > 0) {
                for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                    const Scalar delta = rate_f * wtracer[tIdx];
                    // Injection of free tracer only
                    tr.residual_[tIdx][I][Free] -= delta;

                    // Store _injector_ tracer rate for reporting
                    // (can be done here since WTRACER is constant)
                    tracerRate[tr.idx_[tIdx]].rate += delta;
                    freeTracerRate[tr.idx_[tIdx]].rate += delta;
                    if (eclWell.isMultiSegment()) {
                        (*mswTracerRate)[tr.idx_[tIdx]].rate[eclWell.getConnections().get(i).segment()] += delta;
                    }
                }
                tr.dVol_[Free][I] -= rate_f * dt;
            }
            else if (rate_f < 0) {
                for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                    const Scalar delta = rate_f * wtracer[tIdx];
                    // Store _injector_ tracer rate for cross-flowing well connections
                    // (can be done here since WTRACER is constant)
                    tracerRate[tr.idx_[tIdx]].rate += delta;
                    freeTracerRate[tr.idx_[tIdx]].rate += delta;

                    // Production of free tracer
                    tr.residual_[tIdx][I][Free] -= rate_f * tr.concentration_[tIdx][I][Free];
                }
                tr.dVol_[Free][I] -= rate_f * dt;

                // Derivative matrix for free tracer producer
                system_.diagonal(I)[Free][Free] -= rate_f;
            }
            if (rate_s < 0) {
                for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                    // Production of solution tracer
                    tr.residual_[tIdx][I][Solution] -= rate_s * tr.concentration_[tIdx][I][Solution];
                }
                tr.dVol_[Solution][I] -= rate_s * dt;

                // Derivative matrix for solution tracer producer
                system_.diagonal(I)[Solution][Solution] -= rate_s;
            }
        }
    }

    /*!
     * \brief Boundary conditions for the tracer equations.
     *
     * The flow of the tracer phase across a boundary face of type RATE, FREE or
     * DIRICHLET (BCCON + BCPROP) is the one given by the flow equations, i.e. the
     * boundary flux of the local residual evaluated at the current state. The
     * concentration of the fluid entering through the face is taken from BCTRACER,
     * and fluid leaving the cell carries the tracer concentration of the cell. As
     * for BCPROP, a positive rate is a flow out of the domain and a negative rate
     * is a flow into the domain. Other boundary types carry no mass flow.
     *
     * For oil and gas tracers, outflow is split in a free and a solution part, as for
     * producer wells.
     */
    template<class TrRe>
    void assembleTracerEquationBoundary(TrRe& tr,
                                        const unsigned I,
                                        const Scalar dt)
    {
        const auto& problem = simulator_.problem();
        if (!problem.nonTrivialBoundaryConditions()) {
            return;
        }

        const auto boundaryFaces = system_.boundaryFaces(I);
        if (boundaryFaces.begin() == boundaryFaces.end()) {
            return;
        }

        // Bind the intensive quantities, as the model may return them by value.
        const auto& intQuants = simulator_.model().intensiveQuantities(I, /*timeIdx=*/0);
        const auto& bcstate = simulator_.vanguard().schedule()[problem.episodeIndex()].bcstate;
        for (const auto& bf : boundaryFaces) {
            if (bf.dirId < 0) { // not for NNCs
                continue;
            }

            const int bcIndex = problem.boundaryConditionIndex(I, bf.dirId);
            if (bcIndex == 0) {
                continue;
            }

            const auto [type, massRate] = problem.boundaryCondition(I, bf.dirId);
            if (type != BCType::RATE && type != BCType::FREE && type != BCType::DIRICHLET) {
                continue;
            }

            // Surface volume rate out of the cell of the main component of each phase (water
            // in the water phase etc.). The rate of the tracer phase is zero if the BCPROP
            // component is not the phase of the tracer.
            const auto componentRates = boundaryComponentRates_(type, massRate, I, bf, intQuants);
            const Scalar boundaryArea = bf.area * decay<Scalar>(intQuants.extrusionFactor());
            const auto phaseFlux = [&](const int phaseIdx)
            {
                return componentRates[phaseIdx] * boundaryArea;
            };
            const Scalar rateOut = phaseFlux(tr.phaseIdx_);
            if (rateOut == Scalar{0}) {
                continue;
            }

            // Outflow is split in a free part and a solution part (vaporized oil in the gas
            // phase for oil tracers, dissolved gas in the oil phase for gas tracers), as for
            // producer wells. The component fluxes are F_o + Rv F_g (oil) and F_g + Rs F_o
            // (gas) in terms of the free phase fluxes F_o and F_g. Inflow is free only.
            Scalar rateFree = rateOut;
            Scalar rateSol = 0.0;
            const bool oilSolution = tr.phaseIdx_ == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil();
            const bool gasSolution = tr.phaseIdx_ == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas();
            if (rateOut > 0 && (oilSolution || gasSolution)) {
                const auto& fs = intQuants.fluidState();
                const Scalar Rs = FluidSystem::enableDissolvedGas() ? decay<Scalar>(fs.Rs()) : Scalar{0};
                const Scalar Rv = FluidSystem::enableVaporizedOil() ? decay<Scalar>(fs.Rv()) : Scalar{0};
                const Scalar Qo = phaseFlux(FluidSystem::oilPhaseIdx);
                const Scalar Qg = phaseFlux(FluidSystem::gasPhaseIdx);
                const Scalar det = 1.0 - Rs * Rv;
                const Scalar Fo = (Qo - Rv * Qg) / det;
                const Scalar Fg = (Qg - Rs * Qo) / det;
                if (oilSolution) {
                    rateFree = Fo;
                    rateSol = Rv * Fg;
                }
                else {
                    rateFree = Fg;
                    rateSol = Rs * Fo;
                }
            }

            if (rateOut < 0) {
                // Inflow: concentration from BCTRACER, no tracer if not specified
                addFaceFlux_<Free>(tr, I, FaceFlux{rateFree, false}, dt,
                                   [&](const int tIdx)
                                   {
                                       return bcstate.tracerConcentration(bcIndex, this->name(tr.idx_[tIdx]))
                                                     .value_or(0.0);
                                   });
            }
            else {
                // Outflow: tracer leaves with the concentrations of the cell
                const auto noInflow = [](const int) { return Scalar{0}; };
                addFaceFlux_<Free>(tr, I, FaceFlux{rateFree, true}, dt, noInflow);
                addFaceFlux_<Solution>(tr, I, FaceFlux{rateSol, true}, dt, noInflow);
            }
        }
    }

    /*!
     * \brief Surface volume rate per area out of cell I across a boundary face, of the main
     *        component of each phase (indexed by the phase).
     *
     * This is the boundary flux of the flow equations, as in the boundary rate vector of
     * the problem: the rate given by BCPROP for RATE faces, and the advective flux with the
     * fluid state of the boundary for FREE and DIRICHLET faces, including the dissolved and
     * vaporized components. A positive rate is a flow out of the domain.
     */
    template<class RateVector>
    std::array<Scalar, numPhases>
    boundaryComponentRates_(const BCType type,
                            const RateVector& massRate,
                            const unsigned I,
                            const BoundaryFace& bf,
                            const IntensiveQuantities& intQuants) const
    {
        const auto& problem = simulator_.problem();
        std::array<Scalar, numPhases> rates{};

        if (type == BCType::RATE) {
            // Mass rate, converted to surface volume
            const unsigned pvtRegionIdx = problem.pvtRegionIndex(I);
            for (unsigned phaseIdx = 0; phaseIdx < numPhases; ++phaseIdx) {
                if (!FluidSystem::phaseIsActive(phaseIdx)) {
                    continue;
                }
                const unsigned activeCompIdx =
                    FluidSystem::canonicalToActiveCompIdx(FluidSystem::solventComponentIndex(phaseIdx));
                rates[phaseIdx] = decay<Scalar>(massRate[activeCompIdx]) /
                                  FluidSystem::referenceDensity(phaseIdx, pvtRegionIdx);
            }
            return rates;
        }

        // FREE and DIRICHLET: the phase is carried with the fluid state of the upstream side
        const auto exFluidState = problem.boundaryFluidState(I, bf.dirId);
        std::array<short, numPhases> upIdx{};
        std::array<short, numPhases> dnIdx{};
        std::array<Evaluation, numPhases> volumeFlux{};
        std::array<Evaluation, numPhases> pressureDifference{};
        ExtensiveQuantities::calculateBoundaryGradients_(problem,
                                                         I,
                                                         intQuants,
                                                         bf.bfIdx,
                                                         bf.area,
                                                         bf.zCoord,
                                                         exFluidState,
                                                         upIdx,
                                                         dnIdx,
                                                         volumeFlux,
                                                         pressureDifference);

        const auto& inFs = intQuants.fluidState();
        const unsigned pvtRegionIdx = intQuants.pvtRegionIndex();
        const auto addPhaseRates = [&rates, pvtRegionIdx](const unsigned phaseIdx,
                                                          const Scalar volumeRate,
                                                          const auto& upFs)
        {
            using UpFluidState = std::remove_cvref_t<decltype(upFs)>;
            const Scalar q = getInvB_<FluidSystem, UpFluidState, Scalar>(upFs, phaseIdx, pvtRegionIdx) *
                             volumeRate;
            rates[phaseIdx] += q;
            if (phaseIdx == FluidSystem::oilPhaseIdx && FluidSystem::enableDissolvedGas()) {
                rates[FluidSystem::gasPhaseIdx] +=
                    BlackOil::getRs_<FluidSystem, UpFluidState, Scalar>(upFs, pvtRegionIdx) * q;
            }
            else if (phaseIdx == FluidSystem::waterPhaseIdx && FluidSystem::enableDissolvedGasInWater()) {
                rates[FluidSystem::gasPhaseIdx] +=
                    BlackOil::getRsw_<FluidSystem, UpFluidState, Scalar>(upFs, pvtRegionIdx) * q;
            }
            else if (phaseIdx == FluidSystem::gasPhaseIdx) {
                if (FluidSystem::enableVaporizedOil()) {
                    rates[FluidSystem::oilPhaseIdx] +=
                        BlackOil::getRv_<FluidSystem, UpFluidState, Scalar>(upFs, pvtRegionIdx) * q;
                }
                if (FluidSystem::enableVaporizedWater()) {
                    rates[FluidSystem::waterPhaseIdx] +=
                        BlackOil::getRvw_<FluidSystem, UpFluidState, Scalar>(upFs, pvtRegionIdx) * q;
                }
            }
        };

        for (unsigned phaseIdx = 0; phaseIdx < numPhases; ++phaseIdx) {
            if (!FluidSystem::phaseIsActive(phaseIdx)) {
                continue;
            }
            const Scalar pBoundary = decay<Scalar>(exFluidState.pressure(phaseIdx));
            const Scalar pInside = decay<Scalar>(inFs.pressure(phaseIdx));
            const Scalar volumeRate = decay<Scalar>(volumeFlux[phaseIdx]);
            if (pBoundary < pInside) {
                addPhaseRates(phaseIdx, volumeRate, inFs); // outflux
            }
            else if (pBoundary > pInside) {
                addPhaseRates(phaseIdx, volumeRate, exFluidState); // influx
            }
        }
        return rates;
    }

    template<class TrRe>
    void assembleTracerEquationSource(TrRe& tr,
                                      const Scalar dt,
                                      unsigned I)
    {
        if (tr.numTracer() == 0) {
            return;
        }

        // Skip if solution tracers do not exist
        if (tr.phaseIdx_ ==  FluidSystem::waterPhaseIdx ||
            (tr.phaseIdx_ ==  FluidSystem::gasPhaseIdx && !FluidSystem::enableDissolvedGas()) ||
            (tr.phaseIdx_ ==  FluidSystem::oilPhaseIdx && !FluidSystem::enableVaporizedOil()))
        {
            return;
        }

        const Scalar& dsVol = tr.dVol_[Solution][I];
        const Scalar& dfVol = tr.dVol_[Free][I];

        // Source term determined by sign of dsVol: if dsVol > 0 then ms -> mf, else mf -> ms
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            if (dsVol >= 0) {
                const auto delta = (dfVol / dt) * tr.concentration_[tIdx][I][Free];
                tr.residual_[tIdx][I][Free] -= delta;
                tr.residual_[tIdx][I][Solution] += delta;
            }
            else {
                const auto delta = (dsVol / dt) * tr.concentration_[tIdx][I][Solution];
                tr.residual_[tIdx][I][Free] += delta;
                tr.residual_[tIdx][I][Solution] -= delta;
            }
        }

        // Derivative matrix
        auto& diag = system_.diagonal(I);
        if (dsVol >= 0) {
            const auto delta = dfVol / dt;
            diag[Free][Free] -= delta;
            diag[Solution][Free] += delta;
        }
        else {
            const auto delta = dsVol / dt;
            diag[Free][Solution] += delta;
            diag[Solution][Solution] -= delta;
        }
    }

    //! \brief Reset the well tracer rates before assembling the tracer equations.
    void prepareWellTracerRates_()
    {
        this->wellTracerRate_.clear();
        this->wellFreeTracerRate_.clear();
        this->wellSolTracerRate_.clear();

        // educated guess for new container size
        const auto num_msw = this->mSwTracerRate_.size();
        this->mSwTracerRate_.clear();

        const auto& wellPtrs = simulator_.problem().wellModel().localNonshutWells();
        this->wellTracerRate_.reserve(wellPtrs.size());
        this->wellFreeTracerRate_.reserve(wellPtrs.size());
        this->wellSolTracerRate_.reserve(wellPtrs.size());
        this->mSwTracerRate_.reserve(num_msw);
        for (const auto& wellPtr : wellPtrs) {
            // Resize vectors of well tracer rates to total number of tracers
            const auto& eclWell = wellPtr->wellEcl();
            this->wellTracerRate_[eclWell.seqIndex()].resize(this->numTracers());
            this->wellFreeTracerRate_[eclWell.seqIndex()].resize(this->numTracers());
            this->wellSolTracerRate_[eclWell.seqIndex()].resize(this->numTracers());
            if (eclWell.isMultiSegment()) {
                this->mSwTracerRate_[eclWell.seqIndex()].resize(this->numTracers());
            }
        }
    }

    /*!
     * \brief Assemble the equations of the tracers of a batch.
     *
     * The equations of the batches are independent, so they share the matrix and are
     * assembled and solved one batch at a time.
     */
    template<class TrRe>
    void assembleTracerEquations_(TrRe& tr)
    {
        // Note that we formulate the equations in terms of a concentration update
        // (compared to previous time step) and not absolute concentration.
        // This implies that current concentration (tr.concentration_[][]) contributes
        // to the rhs both through storage and flux terms.
        // Compare also advanceTracerFields(...) below.

        DeferredLogger local_deferredLogger{};
        OPM_BEGIN_PARALLEL_TRY_CATCH()
        {
            OPM_TIMEBLOCK(tracerAssemble);
            system_.clearMatrix();
            for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                tr.residual_[tIdx] = 0.0;
            }

            // Well terms
            for (const auto& wellPtr : simulator_.problem().wellModel().localNonshutWells()) {
                this->assembleTracerEquationWell(tr, *wellPtr);
            }

            this->assembleTracerEquationAquifer_(tr);

            const Scalar dt = simulator_.timeStepSize();
            const bool enableStorageCache = simulator_.model().enableStorageCache();
            const unsigned numCells = system_.numCells();

            // This is safe as each cell I only writes to its own row of the residual and
            // the matrix, and to the blocks (J, I) of its neighbors J.
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (unsigned I = 0; I < numCells; ++I) {
                if (!system_.isInterior(I)) {
                    // Dirichlet boundary conditions for parallel matrix
                    auto& diag = system_.diagonal(I);
                    diag[Free][Free] = 1.;
                    diag[Solution][Solution] = 1.;
                    continue;
                }

                const Scalar scvVolume = simulator_.model().dofTotalVolume(I);
                this->assembleTracerEquationVolume(tr, scvVolume, dt, I, enableStorageCache);

                for (const auto& nbInfo : system_.neighbors(I)) {
                    this->assembleTracerEquationFlux(tr, I, nbInfo, dt);
                }

                // Boundary conditions (BCCON + BCPROP + BCTRACER)
                this->assembleTracerEquationBoundary(tr, I, dt);

                // Source terms (mass transfer between free and solution tracer), which use
                // the total change of the free and solution volume of the cell
                this->assembleTracerEquationSource(tr, dt, I);
            }
        }
        OPM_END_PARALLEL_TRY_CATCH_LOG(local_deferredLogger,
                                       "assembleTracerEquations() failed: ",
                                       true, simulator_.gridView().comm())

        // Communicate overlap using grid Communication
        auto handle = VectorVectorDataHandle<GridView, std::vector<TracerVector>>(tr.residual_,
                                                                                  simulator_.gridView());
        simulator_.gridView().communicate(handle, Dune::InteriorBorder_All_Interface,
                                          Dune::ForwardCommunication);
    }

    template<TracerTypeIdx Index, class TrRe>
    void updateElem(TrRe& tr,
                    const Scalar scvVolume,
                    const unsigned globalDofIdx)
    {
        const Scalar vol1 = computeVolume_<Index>(tr.phaseIdx_, globalDofIdx, 0);
        vol1_[Index][tr.phaseIdx_][globalDofIdx] = vol1 * scvVolume;
        tr.dVol_[Index][globalDofIdx] = 0.0;
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            tr.storageOfTimeIndex1_[tIdx][globalDofIdx][Index] =
                vol1 * tr.concentrationInitial_[tIdx][globalDofIdx][Index];
        }
    }

    void updateStorageCache()
    {
        for (auto& tr : tbatch) {
            if (tr.numTracer() != 0) {
                tr.concentrationInitial_ = tr.concentration_;
            }
        }

        const unsigned numCells = simulator_.model().numGridDof();

        // This is safe as each cell only writes to its own entries in the shared arrays.
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (unsigned globalDofIdx = 0; globalDofIdx < numCells; ++globalDofIdx) {
            const Scalar scvVolume = simulator_.model().dofTotalVolume(globalDofIdx);
            for (auto& tr : tbatch) {
                if (tr.numTracer() == 0) {
                    continue;
                }
                updateElem<Free>(tr, scvVolume, globalDofIdx);
                updateElem<Solution>(tr, scvVolume, globalDofIdx);
            }
        }
    }

    template<TracerTypeIdx Index, class TrRe>
    void copyForOutput(TrRe& tr,
                       const std::vector<TracerVector>& dx,
                       const Scalar S,
                       const unsigned tIdx,
                       const unsigned globalDofIdx,
                       std::vector<TracerVectorSingle>& sc)
    {
        constexpr Scalar tol_gas_sat = 1e-6;
        tr.concentration_[tIdx][globalDofIdx][Index] -= dx[tIdx][globalDofIdx][Index];
        if (tr.concentration_[tIdx][globalDofIdx][Index] < 0.0 || S < tol_gas_sat) {
            tr.concentration_[tIdx][globalDofIdx][Index] = 0.0;
        }
        sc[tr.idx_[tIdx]][globalDofIdx] = tr.concentration_[tIdx][globalDofIdx][Index];
    }

    template<TracerTypeIdx Index, class TrRe>
    void assignRates(const TrRe& tr,
                     const Well& eclWell,
                     const std::size_t i,
                     const std::size_t I,
                     const Scalar rate,
                     std::vector<WellTracerRate<Scalar>>& tracerRate,
                     std::vector<MSWellTracerRate<Scalar>>* mswTracerRate,
                     std::vector<WellTracerRate<Scalar>>& splitRate)
    {
        if (rate < 0) {
            for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                // Store _producer_ free tracer rate for reporting
                const Scalar delta = rate * tr.concentration_[tIdx][I][Index];
                tracerRate[tr.idx_[tIdx]].rate += delta;
                splitRate[tr.idx_[tIdx]].rate += delta;
                if (eclWell.isMultiSegment()) {
                    (*mswTracerRate)[tr.idx_[tIdx]].rate[eclWell.getConnections().get(i).segment()] += delta;
                }
            }
        }
    }

    void convertEffectiveRatesToRawRates() {
        const auto& wellPtrs = simulator_.problem().wellModel().localNonshutWells();
        for (const auto& wellPtr : wellPtrs) {
            const auto& eclWell = wellPtr->wellEcl();
            const auto well_seq_index = eclWell.seqIndex();
            const auto inv_well_eff_factor = 1.0 / std::max(Scalar{1.0e-10}, wellPtr->wellEfficiencyFactor());

            std::ranges::for_each(this->wellTracerRate_[well_seq_index], [&](WellTracerRate<Scalar>& wtr) {
                wtr.rate *= inv_well_eff_factor;
            });
            std::ranges::for_each(this->wellFreeTracerRate_[well_seq_index], [&](WellTracerRate<Scalar>& wtr) {
                wtr.rate *= inv_well_eff_factor;
            });
            std::ranges::for_each(this->wellSolTracerRate_[well_seq_index], [&](WellTracerRate<Scalar>& wtr) {
                wtr.rate *= inv_well_eff_factor;
            });
            if (eclWell.isMultiSegment()) {
                std::ranges::for_each(this->mSwTracerRate_[well_seq_index], [&](MSWellTracerRate<Scalar>& wtr) {
                    std::ranges::for_each(wtr.rate, [&](auto& item) {
                        item.second *= inv_well_eff_factor;
                    });
                });
            }
        }
    }

    /*!
     * \brief Cross-reference the boundary conditions of the schedule with the tracers.
     *
     * This can not be done when parsing the schedule since the tracers are not known
     * there. A BCTRACER entry for an undeclared tracer is an error. The phase of a
     * BCTRACER entry is the BCPROP component of the face, so a warning is given if that
     * differs from the phase of the tracer (the tracer will then not enter).
     *
     * Tracers are transported across boundaries of type RATE, FREE and DIRICHLET. A
     * warning is given for BCTRACER entries on faces that are never of one of these
     * types (they are ignored).
     */
    void checkBoundaryTracers_(DeferredLogger& deferredLogger) const
    {
        std::unordered_map<std::string, int> tracer_name_to_idx;
        for (int tracerIdx = 0; tracerIdx < this->numTracers(); ++tracerIdx) {
            tracer_name_to_idx.emplace(this->name(tracerIdx), tracerIdx);
        }

        const bool logging = simulator_.vanguard().grid().comm().rank() == 0;
        const auto& schedule = simulator_.vanguard().schedule();
        std::set<std::pair<int, std::string>> warned;
        std::set<int> flowIndices; // RATE, FREE or DIRICHLET
        std::set<int> tracerIndices;
        for (std::size_t step = 0; step < schedule.size(); ++step) {
            for (const auto& face : schedule[step].bcstate) {
                if (face.bctype == BCType::RATE || face.bctype == BCType::FREE ||
                    face.bctype == BCType::DIRICHLET) {
                    flowIndices.insert(face.index);
                }

                for (const auto& tracerbc : face.tracerbcvalues) {
                    tracerIndices.insert(face.index);
                    const auto pos = tracer_name_to_idx.find(tracerbc.tracer);
                    if (pos == tracer_name_to_idx.end()) {
                        throw std::runtime_error(fmt::format("BCTRACER: tracer {} (INDEX {}) is not declared in TRACER",
                                                             tracerbc.tracer, face.index));
                    }

                    const int phaseIdx = this->tracerPhaseIdx_[pos->second];
                    const bool samePhase =
                        (face.component == BCComponent::WATER && phaseIdx == FluidSystem::waterPhaseIdx) ||
                        (face.component == BCComponent::OIL && phaseIdx == FluidSystem::oilPhaseIdx) ||
                        (face.component == BCComponent::GAS && phaseIdx == FluidSystem::gasPhaseIdx);
                    // Faces without a BCPROP component (yet), or with a non-phase component, are not checked
                    const bool isPhaseComponent = face.component == BCComponent::WATER ||
                                                  face.component == BCComponent::OIL ||
                                                  face.component == BCComponent::GAS;
                    if (isPhaseComponent && !samePhase &&
                        warned.emplace(face.index, tracerbc.tracer).second && logging)
                    {
                        deferredLogger.warning(fmt::format("BCTRACER: the BCPROP component of INDEX {} is not the phase "
                                                           "of tracer {}, so the tracer will not enter through this "
                                                           "boundary", face.index, tracerbc.tracer));
                    }
                }
            }
        }

        for (const int index : tracerIndices) {
            if (!flowIndices.contains(index) && logging) {
                deferredLogger.warning(fmt::format("BCTRACER: INDEX {} has no BCPROP of type RATE, FREE or DIRICHLET. "
                                                   "BCTRACER is only used for boundaries with a flow and is ignored.",
                                                   index));
            }
        }
    }

    void buildAquiferTracerConnections_(DeferredLogger& deferredLogger)
    {
        aquifer_tracer_cells_.clear();

        const auto& aquifer_cfg = this->eclState_.aquifer();
        if (!aquifer_cfg.active()) {
            return;
        }

        const auto& specs = aquifer_cfg.aquiferTracers();
        if (specs.empty()) {
            return;
        }

        std::unordered_map<std::string, int> tracer_name_to_idx;
        for (int tracerIdx = 0; tracerIdx < this->numTracers(); ++tracerIdx) {
            tracer_name_to_idx.emplace(this->name(tracerIdx), tracerIdx);
        }

        const auto& vanguard = simulator_.vanguard();
        for (const auto& spec : specs) {
            if (!aquifer_cfg.hasAnalyticalAquifer(spec.aquifer_id)) {
                continue;
            }

            const auto tracer_pos = tracer_name_to_idx.find(spec.tracer_name);
            if (tracer_pos == tracer_name_to_idx.end()) {
                if (simulator_.vanguard().grid().comm().rank() == 0) {
                    deferredLogger.warning(fmt::format("AQANTRC tracer '{}' is not declared in TRACER",
                                                       spec.tracer_name));
                }
                continue;
            }

            if (!aquifer_cfg.connections().hasAquiferConnections(spec.aquifer_id)) {
                continue;
            }

            const int tracerIdx = tracer_pos->second;
            const int phaseIdx = this->tracerPhaseIdx_[tracerIdx];

            for (const auto& conn : aquifer_cfg.connections().getConnections(spec.aquifer_id)) {
                const int cellIdx = vanguard.compressedIndex(conn.global_index);
                if (cellIdx < 0) {
                    continue;
                }

                aquifer_tracer_cells_[cellIdx].push_back(
                    AquiferTracerCellSpec { tracerIdx, phaseIdx, static_cast<Scalar>(spec.concentration) });
            }
        }
    }

    template<class TrRe>
    void assembleTracerEquationAquifer_(TrRe& tr)
    {
        if (tr.numTracer() == 0 || aquifer_tracer_cells_.empty()) {
            return;
        }

        const auto& aquiferModel = simulator_.problem().aquiferModel();
        const Scalar dt = simulator_.timeStepSize();

        for (const auto& [cellIdx, specs] : aquifer_tracer_cells_) {
            // Scalar influx from converged Qai_; do not recalculate or use getValue()
            // in the flow Newton path (see AquiferAnalytical::addToSource).
            const Scalar rate = aquiferModel.cachedConnectionInfluxRate(cellIdx);
            if (rate == Scalar{0}) {
                continue;
            }

            const unsigned I = cellIdx;
            const Scalar rate_f = rate;

            for (const auto& spec : specs) {
                if (spec.phaseIdx != tr.phaseIdx_) {
                    continue;
                }

                int localTIdx = -1;
                for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                    if (tr.idx_[tIdx] == spec.tracerIdx) {
                        localTIdx = tIdx;
                        break;
                    }
                }
                if (localTIdx < 0) {
                    continue;
                }

                if (rate_f > 0) {
                    const Scalar delta = rate_f * spec.concentration;
                    tr.residual_[localTIdx][I][Free] -= delta;
                    tr.dVol_[Free][I] -= rate_f * dt;
                }
                else if (rate_f < 0) {
                    tr.residual_[localTIdx][I][Free] -= rate_f * tr.concentration_[localTIdx][I][Free];
                    tr.dVol_[Free][I] -= rate_f * dt;
                    system_.diagonal(I)[Free][Free] -= rate_f;
                }
            }
        }
    }

    template<class TrRe>
    bool solveBatch_(TrRe& tr, std::vector<TracerVector>& dx)
    {
        // Istl solver
        if (!this->useReorderSolver_) {
            return this->linearSolveBatchwise_(system_.matrix(), dx, tr.residual_);
        }

        // Run istl solver fallback for some number of time steps before retrying reorder
        if (tr.reorderSkipSteps_ > 0) {
            --tr.reorderSkipSteps_;
            if (tr.reorderSkipSteps_ == 0) {
                OpmLog::info(fmt::format("Tracer model: retrying the reorder solver for the {} "
                                         "tracers in the next time step",
                                         FluidSystem::phaseName(tr.phaseIdx_)));
            }
            return this->linearSolveBatchwise_(system_.matrix(), dx, tr.residual_);
        }

        // Reorder solver
        if (this->linearSolveReorder_(system_.matrix(), dx, tr.residual_)) {
            return true;
        }

        // Initiate fallback istl solver, which will be done for some number of future time steps
        // as well
        tr.reorderSkipSteps_ = reorderRetrySteps_;
        OpmLog::warning(fmt::format("Tracer model: the reorder solver failed for the {} tracers. "
                                    "Using the ISTL solver for this and the next {} time steps.",
                                    FluidSystem::phaseName(tr.phaseIdx_), reorderRetrySteps_));
        return this->linearSolveBatchwise_(system_.matrix(), dx, tr.residual_);
    }

    void advanceTracerFields()
    {
        prepareWellTracerRates_();

        for (auto& tr : tbatch) {
            if (tr.numTracer() == 0) {
                continue;
            }

            assembleTracerEquations_(tr);

            // Note that we solve for a concentration update (compared to previous time step)
            // Confer also assembleTracerEquations_(...) above.
            std::vector<TracerVector> dx(tr.concentration_);
            for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                dx[tIdx] = 0.0;
            }

            const bool converged = solveBatch_(tr, dx);
            if (!converged) {
                OpmLog::warning("### Tracer model: Linear solver did not converge. ###");
            }

            OPM_TIMEBLOCK(tracerPost);

            for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                for (std::size_t globalDofIdx = 0; globalDofIdx < tr.concentration_[tIdx].size(); ++globalDofIdx) {
                    // New concetration. Concentrations that are negative or where free/solution phase is not
                    // present are set to zero
                    const auto& intQuants = simulator_.model().intensiveQuantities(globalDofIdx, 0);
                    const auto& fs = intQuants.fluidState();
                    const Scalar Sf = decay<Scalar>(fs.saturation(tr.phaseIdx_));
                    Scalar Ss = 0.0;

                    if (tr.phaseIdx_ == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                        Ss = decay<Scalar>(fs.saturation(FluidSystem::oilPhaseIdx));
                    }
                    else if (tr.phaseIdx_ == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                        Ss = decay<Scalar>(fs.saturation(FluidSystem::gasPhaseIdx));
                    }

                    copyForOutput<Free>(tr, dx, Sf, tIdx, globalDofIdx, this->freeTracerConcentration_);
                    copyForOutput<Solution>(tr, dx, Ss, tIdx, globalDofIdx, this->solTracerConcentration_);
                }
            }

            // Store _producer_ tracer rate for reporting
            const auto& wellPtrs = simulator_.problem().wellModel().localNonshutWells();
            for (const auto& wellPtr : wellPtrs) {
                const auto& eclWell = wellPtr->wellEcl();

                // Injection rates already reported during assembly
                if (!eclWell.isProducer()) {
                    continue;
                }

                Scalar rateWellPos = 0.0;
                Scalar rateWellNeg = 0.0;
                const std::size_t well_index = simulator_.problem().wellModel().wellState().index(eclWell.name()).value();
                const auto& ws = simulator_.problem().wellModel().wellState().well(well_index);
                auto& tracerRate = this->wellTracerRate_[eclWell.seqIndex()];
                auto& freeTracerRate = this->wellFreeTracerRate_[eclWell.seqIndex()];
                auto& solTracerRate = this->wellSolTracerRate_[eclWell.seqIndex()];
                auto* mswTracerRate = eclWell.isMultiSegment() ? &this->mSwTracerRate_[eclWell.seqIndex()] : nullptr;
                const auto well_eff = wellPtr->wellEfficiencyFactor(); // Needed to convert ws.perf_data.phase_mixing_rates to effective rates
                for (std::size_t i = 0; i < ws.perf_data.size(); ++i) {
                    const auto I = ws.perf_data.cell_index[i];
                    const Scalar rate = wellPtr->volumetricSurfaceRateForConnection(I, tr.phaseIdx_); // Includes (accumulated) well efficiency factor

                    Scalar rate_s;
                    if (tr.phaseIdx_ == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                        rate_s = ws.perf_data.phase_mixing_rates[i][ws.vaporized_oil]*well_eff;
                    }
                    else if (tr.phaseIdx_ == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                        rate_s = ws.perf_data.phase_mixing_rates[i][ws.dissolved_gas]*well_eff;
                    }
                    else {
                        rate_s = 0.0;
                    }

                    const Scalar rate_f = rate - rate_s;
                    assignRates<Free>(tr, eclWell, i, I, rate_f,
                                      tracerRate, mswTracerRate, freeTracerRate);
                    assignRates<Solution>(tr, eclWell, i, I, rate_s,
                                          tracerRate, mswTracerRate, solTracerRate);

                    if (rate < 0) {
                        rateWellNeg += rate;
                    }
                    else {
                        rateWellPos += rate;
                    }
                }

                // TODO: Some inconsistencies here that perhaps should be clarified.
                // The "offical" rate as reported below is occasionally significant
                // different from the sum over connections (as calculated above). Only observed
                // for small values, neglible for the rate itself, but matters when used to
                // calculate tracer concentrations.
                const Scalar official_well_rate_total =
                    simulator_.problem().wellModel().wellState().well(well_index).surface_rates[tr.phaseIdx_];

                const Scalar rateWellTotal = official_well_rate_total;

                if (rateWellTotal > rateWellNeg) { // Cross flow
                    constexpr Scalar bucketPrDay = 10.0 / (1000. * 3600. * 24.); // ... keeps (some) trouble away
                    const Scalar factor = (rateWellTotal < -bucketPrDay) ? rateWellTotal / rateWellNeg : 0.0;
                    for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                        tracerRate[tr.idx_[tIdx]].rate *= factor;
                    }
                }
            }
        }
        convertEffectiveRatesToRawRates();
    }

    Simulator& simulator_;

    // True if DISPERC is given in the deck
    bool dispersionActive_;

    // This struct collects tracers of the same type (i.e, transported in same phase).
    // The idea being that, under the assumption of linearity, tracers of same type can
    // be solved in concert, having a common system matrix but separate right-hand-sides.

    // Since oil or gas tracers appears in dual compositions when VAPOIL respectively DISGAS
    // is active, the template argument is intended to support future extension to these
    // scenarios by supplying an extended vector type.

    template <typename TV>
    struct TracerBatch
    {
        std::vector<int> idx_;
        const int phaseIdx_;
        std::vector<TV> concentrationInitial_;
        std::vector<TV> concentration_;
        std::vector<TV> storageOfTimeIndex1_;
        std::vector<TV> residual_;
        int reorderSkipSteps_ = 0;
        // Molecular diffusion coefficients [m2/s] of the free tracer and of the tracer in solution
        // (TRDIFxxx), common to all tracers in the batch
        Scalar freeDiffusionCoefficient_ = 0.0;
        Scalar solutionDiffusionCoefficient_ = 0.0;
        // Change of the free and solution volume of the phase of the batch in the current time step
        std::array<std::vector<Scalar>, 2> dVol_;

        bool operator==(const TracerBatch& rhs) const
        {
            return this->concentrationInitial_ == rhs.concentrationInitial_ &&
                   this->concentration_ == rhs.concentration_;
        }

        static TracerBatch serializationTestObject()
        {
            TracerBatch<TV> result(4);
            result.idx_ = {1,2,3};
            result.concentrationInitial_ = {5.0, 6.0};
            result.concentration_ = {7.0, 8.0};
            result.storageOfTimeIndex1_ = {9.0, 10.0, 11.0};
            result.residual_ = {12.0, 13.0};

            return result;
        }

        template<class Serializer>
        void serializeOp(Serializer& serializer)
        {
            serializer(concentrationInitial_);
            serializer(concentration_);
        }

        TracerBatch(int phaseIdx = 0) : phaseIdx_(phaseIdx) {}

        int numTracer() const
        { return idx_.size(); }

        void addTracer(const int idx, const TV& concentration)
        {
            const int numGridDof = concentration.size();
            idx_.emplace_back(idx);
            concentrationInitial_.emplace_back(concentration);
            concentration_.emplace_back(concentration);
            residual_.emplace_back(numGridDof);
            storageOfTimeIndex1_.emplace_back(numGridDof);
            for (auto& dVol : dVol_) {
                dVol.resize(numGridDof);
            }
        }
    };

    std::vector<TracerBatch<TracerVector>> tbatch;
    std::array<std::array<std::vector<Scalar>,numPhases>,2> vol1_;
    LinearSystem system_{};

    struct AquiferTracerCellSpec {
        int tracerIdx{};
        int phaseIdx{};
        Scalar concentration{};
    };

    std::unordered_map<unsigned, std::vector<AquiferTracerCellSpec>> aquifer_tracer_cells_;

    int reorderRetrySteps_ = 0;
};

} // namespace Opm

#endif // OPM_TRACER_MODEL_HPP
