/*
  Copyright 2026, SINTEF Digital

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_NONLINEAR_SYSTEM_COMPOSITIONAL_IMPL_HEADER_INCLUDED
#define OPM_NONLINEAR_SYSTEM_COMPOSITIONAL_IMPL_HEADER_INCLUDED

#ifndef OPM_NONLINEAR_SYSTEM_COMPOSITIONAL_HEADER_INCLUDED
#include <config.h>
#include <opm/simulators/flow/NonlinearSystemCompositional.hpp>
#endif

#include <dune/common/timer.hh>

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/OpmLog/OpmLog.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace Opm {

template <class TypeTag>
NonlinearSystemCompositional<TypeTag>::
NonlinearSystemCompositional(Simulator& simulator,
                             const ModelParameters& param,
                             CompWellModel<TypeTag>& wellModel,
                             const bool terminalOutput)
    : ParentType(simulator, param, wellModel, terminalOutput)
{
    this->convergence_reports_.reserve(64);
}

template <class TypeTag>
SimulatorReportSingle
NonlinearSystemCompositional<TypeTag>::
prepareStep(const SimulatorTimerInterface& timer)
{
    SimulatorReportSingle report;
    Dune::Timer perfTimer;
    perfTimer.start();

    const int lastStepFailed = timer.lastStepFailed();
    if (this->grid_.comm().size() > 1
        && this->grid_.comm().max(lastStepFailed) != this->grid_.comm().min(lastStepFailed)) {
        OPM_THROW(std::runtime_error,
                  "Misalignment of the parallel simulation run in prepareStep "
                  "- the previous step succeeded on some ranks but failed on others.");
    }

    if (lastStepFailed) {
        this->wellModel().restoreLastValidState();
        this->simulator_.model().updateFailed();
    }
    else {
        this->simulator_.model().advanceTimeLevel();
    }

    this->simulator_.setTime(timer.simulationTimeElapsed());
    this->simulator_.setTimeStepSize(timer.currentStepLength());

    this->simulator_.problem().resetIterationForNewTimestep();
    this->simulator_.problem().beginTimeStep();

    report.pre_post_time += perfTimer.stop();
    return report;
}

template <class TypeTag>
void
NonlinearSystemCompositional<TypeTag>::
initialLinearization(SimulatorReportSingle& report,
                     const int minIter,
                     const int maxIter,
                     const SimulatorTimerInterface& timer)
{
    ParentType::initialLinearization(report,
                                     minIter,
                                     maxIter,
                                     timer);
    Dune::Timer perfTimer;
    perfTimer.start();

    // Calculate reservoir and well convergence and store in convergence history
    auto convrep = getConvergence(timer);

    // Report converged flag
    report.converged = convrep.converged()
        && this->simulator_.problem().iterationContext().iteration() >= minIter;

    // Throw for severe failures
    const auto severity = convrep.severityOfWorstFailure();
    this->convergence_reports_.back().report.push_back(std::move(convrep));
    if (severity == ConvergenceReport::Severity::NotANumber) {
        this->failureReport_ += report;
        OPM_THROW_PROBLEM(NumericalProblem, "NaN convergence values found!");
    }

    if (severity == ConvergenceReport::Severity::TooLarge) {
        this->failureReport_ += report;
        OPM_THROW_NOLOG(NumericalProblem, "Too large convergence values found!");
    }

    report.update_time += perfTimer.stop();

    // Store residual norms in history container
    const auto residualMetrics = this->reservoirResidualMetrics();
    this->residual_norms_history_.push_back(residualMetrics);

}

template <class TypeTag>
template <class NonlinearSolverType>
SimulatorReportSingle
NonlinearSystemCompositional<TypeTag>::
nonlinearIteration(const SimulatorTimerInterface& timer,
                   NonlinearSolverType& nonlinearSolver)
{
    if (this->simulator_.problem().iterationContext().needsTimestepInit()) {
        this->residual_norms_history_.clear();
        this->current_relaxation_ = 1.0;
        this->dx_old_ = 0.0;
        this->convergence_reports_.push_back({timer.reportStepNum(), timer.currentStepNum(), {}});
        this->convergence_reports_.back().report.reserve(numEq);
    }

    auto result = this->nonlinearIterationNewton(timer, nonlinearSolver);
    this->simulator_.problem().advanceIteration();
    return result;
}

template <class TypeTag>
template <class NonlinearSolverType>
SimulatorReportSingle
NonlinearSystemCompositional<TypeTag>::
nonlinearIterationNewton(const SimulatorTimerInterface& timer,
                         NonlinearSolverType& nonlinearSolver)
{
    OPM_TIMEFUNCTION();

    SimulatorReportSingle report;
    Dune::Timer perfTimer;

    this->initialLinearization(report,
                               this->param_.newton_min_iter_,
                               this->param_.newton_max_iter_,
                               timer);

    if (!report.converged) {
        perfTimer.reset();
        perfTimer.start();
        report.total_newton_iterations = 1;

        BVector x(this->simulator_.model().numGridDof());
        this->linear_solve_setup_time_ = 0.0;

        try {
            auto& linearizer = this->simulator_.model().linearizer();
            linearizer.linearizeAuxiliaryEquations();
            linearizer.finalize();

            this->solveJacobianSystem(x);

            report.linear_solve_setup_time += this->linear_solve_setup_time_;
            report.linear_solve_time += perfTimer.stop();
            report.total_linear_iterations += this->linearIterationsLastSolve();
        }
        catch (...) {
            report.linear_solve_setup_time += this->linear_solve_setup_time_;
            report.linear_solve_time += perfTimer.stop();
            report.total_linear_iterations += this->linearIterationsLastSolve();

            this->failureReport_ += report;
            throw;
        }

        perfTimer.reset();
        perfTimer.start();

        auto& model = this->simulator_.model();
        for (unsigned auxModIdx = 0; auxModIdx < model.numAuxiliaryModules(); ++auxModIdx) {
            model.auxiliaryModule(auxModIdx)->postSolve(x);
        }

        if (this->param_.use_update_stabilization_) {
            bool isOscillate = false;
            bool isStagnate = false;
            nonlinearSolver.detectOscillations(this->residual_norms_history_,
                                               this->residual_norms_history_.size() - 1,
                                               isOscillate,
                                               isStagnate);

            if (isOscillate) {
                this->current_relaxation_ -= nonlinearSolver.relaxIncrement();
                this->current_relaxation_ = std::max(this->current_relaxation_, nonlinearSolver.relaxMax());

                if (this->terminalOutputEnabled()) {
                    OpmLog::info("    Oscillating behavior detected: Relaxation set to "
                                 + std::to_string(this->current_relaxation_));
                }
            }

            nonlinearSolver.stabilizeNonlinearUpdate(x, this->dx_old_, this->current_relaxation_);
        }

        this->updateSolution(x);
        report.update_time += perfTimer.stop();
    }

    return report;
}

template <class TypeTag>
typename NonlinearSystemCompositional<TypeTag>::Scalar
NonlinearSystemCompositional<TypeTag>::
relativeChange() const
{
    Scalar resultDelta = 0.0;
    Scalar resultDenom = 0.0;

    const auto& elemMapper = this->simulator_.model().elementMapper();
    const auto& gridView = this->simulator_.gridView();

    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        const unsigned globalElemIdx = elemMapper.index(elem);
        const auto& priVarsNew = this->simulator_.model().solution(/*timeIdx=*/0)[globalElemIdx];
        const auto& priVarsOld = this->simulator_.model().solution(/*timeIdx=*/1)[globalElemIdx];

        for (int pvIdx = 0; pvIdx < static_cast<int>(priVarsNew.size()); ++pvIdx) {
            const auto delta = priVarsNew[pvIdx] - priVarsOld[pvIdx];
            resultDelta += delta * delta;
            resultDenom += priVarsNew[pvIdx] * priVarsNew[pvIdx];
        }
    }

    resultDelta = gridView.comm().sum(resultDelta);
    resultDenom = gridView.comm().sum(resultDenom);

    return resultDenom > 0.0 ? resultDelta / resultDenom : 0.0;
}

template <class TypeTag>
void
NonlinearSystemCompositional<TypeTag>::
solveJacobianSystem(BVector& x)
{
    auto& jacobian = this->simulator_.model().linearizer().jacobian();
    auto& residual = this->simulator_.model().linearizer().residual();
    auto& linSolver = this->simulator_.model().newtonMethod().linearSolver();

    x = 0.0;

    Dune::Timer perfTimer;
    perfTimer.start();
    linSolver.prepare(jacobian, residual);
    this->linear_solve_setup_time_ = perfTimer.stop();
    linSolver.setResidual(residual);
    linSolver.getResidual(residual);
    linSolver.setMatrix(jacobian);
    linSolver.solve(x);
}

template <class TypeTag>
std::vector<typename NonlinearSystemCompositional<TypeTag>::Scalar>
NonlinearSystemCompositional<TypeTag>::
reservoirResidualMetrics() const
{
    const auto& model = this->simulator_.model();
    const auto& residual = model.linearizer().residual();
    const auto& constraintsMap = model.linearizer().constraintsMap();

    std::vector<Scalar> residualMetrics(numEq, 0.0);

    // Ghost-cell residuals may omit fluxes from neighbors beyond the overlap.
    // Use the owning rank's residual to assess convergence.
    const auto& elemMapper = model.elementMapper();
    for (const auto& elem : elements(this->simulator_.gridView(), Dune::Partitions::interior)) {
        const unsigned dofIdx = elemMapper.index(elem);
        if (dofIdx >= model.numGridDof() || model.dofTotalVolume(dofIdx) <= 0.0) {
            continue;
        }

        if (constraintsMap.count(dofIdx) > 0) {
            continue;
        }

        const auto& localResidual = residual[dofIdx];
        for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
            residualMetrics[eqIdx] = std::max(
                residualMetrics[eqIdx],
                std::abs(localResidual[eqIdx] * model.eqWeight(dofIdx, eqIdx)));
        }
    }

    if (this->grid_.comm().size() > 1 && !residualMetrics.empty()) {
        this->grid_.comm().max(residualMetrics.data(), residualMetrics.size());
    }

    return residualMetrics;
}

template <class TypeTag>
void
NonlinearSystemCompositional<TypeTag>::
prepareSolutionUpdate()
{
    // Init. solution update vector
    unsigned nc = this->simulator_.model().numGridDof();
    dP_.resize(nc);
    dSeff_.resize(nc);
    dP_ = 0.0;
    dSeff_ = 0.0;
    effSatData_.resize(nc);

    const auto& elemMapper = this->simulator_.model().elementMapper();
    const auto& gridView = this->simulator_.gridView();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
        // Compute effective saturation before Newton iteration
        unsigned globalElemIdx = elemMapper.index(elem);
        // TODO: use element context?
        const auto* intQuants = this->simulator_.model().cachedIntensiveQuantities(globalElemIdx, /*timeIdx=*/0);
        assert(intQuants);
        const auto& fs = intQuants->fluidState();
        effSatData_[globalElemIdx] = computeEffectiveSaturationData_(fs);
    }
}

template <class TypeTag>
void
NonlinearSystemCompositional<TypeTag>::
storeSolutionUpdate(const GlobalEqVector& dx)
{
    const auto& elemMapper = this->simulator_.model().elementMapper();
    const auto& gridView = this->simulator_.gridView();
    for (const auto& elem : elements(gridView, Dune::Partitions::interior)) {
         unsigned globalElemIdx = elemMapper.index(elem);

        // Store pressure update
        const auto& dP = dx[globalElemIdx][Indices::pressure0Idx];
        dP_[globalElemIdx] = dP;

        // Calculate effective saturation after Newton iteration to use in diff.
        // TODO: use element context?
        const auto* intQuants = this->simulator_.model().cachedIntensiveQuantities(globalElemIdx, /*timeIdx=*/0);
        assert(intQuants);
        const auto& fs = intQuants->fluidState();
        dSeff_[globalElemIdx] = effectiveSaturationChange_(effSatData_[globalElemIdx],
                                                           computeEffectiveSaturationData_(fs));
    }
}

template <class TypeTag>
ConvergenceReport
NonlinearSystemCompositional<TypeTag>::
getConvergence(const SimulatorTimerInterface& timer)
{
    // Reservoir compositional convergence report
    auto report = getCompositionalConvergence(timer.simulationTimeElapsed());

    // Well convergence report
    ConvergenceReport wellReport(timer.simulationTimeElapsed());
    const bool wellConverged = this->wellModel().getWellConvergence();
    using CR = ConvergenceReport;
    if (!wellConverged) {
        // Random failure here since CompWellModel does not return ConvergenceReport
        // TODO: change this when compositional
        wellReport.setWellFailed(
            {CR::WellFailure::Type::Unsolvable, CR::Severity::Normal, -1, "Unknown"});
    }

    report += wellReport;
    return report;
}

template <class TypeTag>
ConvergenceReport
NonlinearSystemCompositional<TypeTag>::
getCompositionalConvergence(double reportTime)
{
    // Init. nonlinear iteration convergence report
    ConvergenceReport report{reportTime};

    using CR = ConvergenceReport;
    using FailureType = CR::ReservoirFailure::Type;
    const std::array types = {FailureType::MaxDP, FailureType::MaxDSeff};

    // No solution update exists yet in the first iteration of a timestep
    const auto& iterCtx = this->simulator_.problem().iterationContext();
    const bool hasSolutionUpdate = !iterCtx.isFirstGlobalIteration();

    // Init. (local) max(dP) and max(dS) data
    Scalar dPmax = 0.0;
    Scalar dSmax = 0.0;

    if (hasSolutionUpdate) {
        // Compute local convergence data
        localCompositionalConvergenceData(dPmax, dSmax);

        // Compute global convergence data
        compositionalConvergenceReduction(dPmax, dSmax);

        // Report convergence
        const std::array<Scalar, 2> dSolmax = {dPmax, dSmax};
        const std::array<std::string, 2> dSolnames = {"DPMAX", "DSEFFMAX"};
        const std::array<Scalar, 2> tolerances
            = {this->param_.tolerance_max_dp_, this->param_.tolerance_max_ds_};
        Scalar maxDSolAllowed = 1.0e20;
        addCompositionalConvergenceMetrics(report,
                                           dSolmax,
                                           dSolnames,
                                           types,
                                           tolerances,
                                           maxDSolAllowed,
                                           [this](const std::string& message) {
                                               if (this->terminal_output_) {
                                                   OpmLog::debug(message);
                                               }
                                           });
    }
    else {
        for (const auto type : types) {
            report.setReservoirFailed({type, CR::Severity::Normal, -1});
        }
    }

    // Output convergence
    if (this->terminal_output_) {
        // Header
        if (iterCtx.isFirstGlobalIteration()) {
            std::string msg = "Iter    DPMAX      DSMAX  ";
            OpmLog::debug(msg);
        }

        // Print values
        std::ostringstream ss;
        const std::streamsize oprec = ss.precision(3);
        const std::ios::fmtflags oflags = ss.setf(std::ios::scientific);

        ss << std::setw(4) << iterCtx.iteration();
        if (hasSolutionUpdate) {
            ss << std::setw(11) << dPmax;
            ss << std::setw(11) << dSmax;
        }
        else {
            ss << std::setw(11) << "-";
            ss << std::setw(11) << "-";
        }

        ss.precision(oprec);
        ss.flags(oflags);

        OpmLog::debug(ss.str());
    }

    return report;
}

template <class TypeTag>
void
NonlinearSystemCompositional<TypeTag>::
localCompositionalConvergenceData(Scalar& dPmax, Scalar& dSmax)
{
    // Max. absolute pressure and effective saturation change over (local) cells
    dPmax = dP_.infinity_norm();
    dSmax = dSeff_.infinity_norm();
}

template <class TypeTag>
void
NonlinearSystemCompositional<TypeTag>::
compositionalConvergenceReduction(Scalar& dPmax, Scalar& dSmax)
{
    // Communicate max. values
    dPmax = this->grid_.comm().max(dPmax);
    dSmax = this->grid_.comm().max(dSmax);
}

template <class TypeTag>
template <class LogFailure>
void
NonlinearSystemCompositional<TypeTag>::
addCompositionalConvergenceMetrics(
    ConvergenceReport& report,
    const std::span<const Scalar> dSolmax,
    const std::span<const std::string> dSolnames,
    const std::span<const ConvergenceReport::ReservoirFailure::Type> types,
    const std::span<const Scalar> tolerances,
    const Scalar maxdSolMaxAllowed,
    LogFailure&& logFailure) const
{
    if (dSolmax.size() != dSolnames.size() || dSolmax.size() != types.size()
        || dSolmax.size() != tolerances.size()) {
        OPM_THROW(std::logic_error, "Mismatched compositional convergence metric sizes.");
    }

    using CR = ConvergenceReport;
    for (std::size_t metricIdx = 0; metricIdx < dSolmax.size(); ++metricIdx) {
        const auto dsolmax = dSolmax[metricIdx];
        const auto dsolname = dSolnames[metricIdx];
        const auto type = types[metricIdx];
        const auto tolerance = tolerances[metricIdx];

        // Failures
        if (std::isnan(dsolmax)) {
            report.setReservoirFailed({type, CR::Severity::NotANumber, -1});
            logFailure("NaN value for " + dsolname + " .");
        }
        else if (dsolmax > maxdSolMaxAllowed) {
            report.setReservoirFailed({type, CR::Severity::TooLarge, -1});
            logFailure("Too large value for " + dsolname + " .");
        }
        else if (dsolmax < 0.0) {
            report.setReservoirFailed({type, CR::Severity::Normal, -1});
            logFailure("Negative value for " + dsolname + " .");
        }
        else if (dsolmax > tolerance) {
            report.setReservoirFailed({type, CR::Severity::Normal, -1});
        }

        report.setReservoirConvergenceMetric(type, -1, dsolmax, tolerance);
    }
}

template <class TypeTag>
template <class FluidState>
typename NonlinearSystemCompositional<TypeTag>::EffectiveSaturationData
NonlinearSystemCompositional<TypeTag>::
computeEffectiveSaturationData_(const FluidState& fs) const
{
    EffectiveSaturationData data;

    // Component moles per pore volume
    for (const int phaseIdx : {FluidSystem::oilPhaseIdx, FluidSystem::gasPhaseIdx}) {
        const Scalar Sb = decay<Scalar>(fs.saturation(phaseIdx) * fs.molarDensity(phaseIdx));
        for (int compIdx = 0; compIdx < numComponents; ++compIdx) {
            data.molarDens[compIdx] += Sb * decay<Scalar>(fs.moleFraction(phaseIdx, compIdx));
        }
    }

    // Mixture molar volume and its derivatives w.r.t. z at fixed pressure
    const auto& L = fs.L();
    const auto v = L / fs.molarDensity(FluidSystem::oilPhaseIdx)
        + (1.0 - L) / fs.molarDensity(FluidSystem::gasPhaseIdx);
    data.molarVolume = decay<Scalar>(v);
    for (int compIdx = 0; compIdx < numComponents - 1; ++compIdx) {
        data.z[compIdx] = decay<Scalar>(fs.moleFraction(compIdx));
        data.dMolarVolumeDz[compIdx] = v.derivative(Indices::z0Idx + compIdx);
    }

    // Water volume per pore volume is m_w / rho_w
    if constexpr (waterEnabled) {
        data.waterDensity = decay<Scalar>(fs.density(FluidSystem::waterPhaseIdx));
        data.waterMassDens =
            decay<Scalar>(fs.saturation(FluidSystem::waterPhaseIdx)) * data.waterDensity;
    }

    return data;
}

template <class TypeTag>
typename NonlinearSystemCompositional<TypeTag>::Scalar
NonlinearSystemCompositional<TypeTag>::
effectiveSaturationChange_(const EffectiveSaturationData& oldData,
                           const EffectiveSaturationData& newData)
{
    // Linearized change in fluid volume per pore volume at fixed pressure:
    // dS' = v * dM + sum_j dv/dz_j * (dm_j - z_j * dM) + dm_w / rho_w
    Scalar dTotMolarDens = 0.0;
    for (int compIdx = 0; compIdx < numComponents; ++compIdx) {
        dTotMolarDens += newData.molarDens[compIdx] - oldData.molarDens[compIdx];
    }

    Scalar dSeff = oldData.molarVolume * dTotMolarDens;
    for (int compIdx = 0; compIdx < numComponents - 1; ++compIdx) {
        const Scalar dMolarDens = newData.molarDens[compIdx] - oldData.molarDens[compIdx];
        dSeff += oldData.dMolarVolumeDz[compIdx] * (dMolarDens - oldData.z[compIdx] * dTotMolarDens);
    }

    if constexpr (waterEnabled) {
        if (oldData.waterDensity > 0.0) {
            dSeff += (newData.waterMassDens - oldData.waterMassDens) / oldData.waterDensity;
        }
    }

    return dSeff;
}

} // namespace Opm

#endif // OPM_NONLINEAR_SYSTEM_COMPOSITIONAL_IMPL_HEADER_INCLUDED
