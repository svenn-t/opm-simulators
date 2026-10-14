/*
  Copyright 2026 NORCE Norwegian Research Centre AS

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

#ifndef OPM_UPWIND_TRANSPORT_SOLVER_HEADER_INCLUDED
#define OPM_UPWIND_TRANSPORT_SOLVER_HEADER_INCLUDED

#include <opm/common/utility/SparseTable.hpp>

#include <opm/simulators/utils/Tarjan.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// \file
///
/// Solver for the linear system of an implicit upwind transport scheme, cell
/// by cell in the order of the flow (a "reordering" solver).
///
/// solveReordered() takes an assembled sparse matrix `A` with square N x N
/// blocks, e.g. a `Dune::BCRSMatrix`, and solves `A x_q = b_q`. Row `r`
/// depends on the cells `c` whose off-diagonal block `A[r][c]` is nonzero;
/// for an upwind scheme these are the upwind neighbours of `r`. Blocks that
/// are in the sparsity pattern but zero are ignored, so a pattern with all
/// neighbours can be used. The block of a cell can couple several unknowns,
/// such as the free and the dissolved part of a tracer, which can also flow
/// in different directions.
///
/// All right-hand sides share the matrix, so they are solved together.
///
/// The cells are ordered with Opm::tarjan(). Solving the components in that
/// order means every component only needs values that are already final:
///
///  - A cell that is not part of a flow loop is solved with its diagonal
///    (block) alone.
///  - Cells in a flow loop depend on each other. Small loops are solved
///    directly, large ones with (block) Gauss-Seidel sweeps. The sweeps follow
///    the flow, since the cells of a component come out of Opm::tarjan() with
///    the upstream ones first.
///
/// With non-positive off-diagonal entries and a diagonal that dominates them,
/// as for an upwind scheme with storage, the matrix is an M-matrix. The
/// solution is then non-negative for non-negative `rhs`, and Gauss-Seidel
/// converges.
///
/// This pays off when the matrix is nearly triangular after reordering, as for
/// pure advection. With diffusion or conduction all neighbours depend on each
/// other, the whole domain becomes one loop, and an iterative solver is the
/// better choice.

namespace Opm
{

struct UpwindTransportSettings
{
    /// Cells in a flow loop are solved directly if there are at most this
    /// many of them, and by iteration if there are more.
    int maxDirectSize = 100;

    /// The iteration stops when the largest change of a quantity in a sweep
    /// is at most `tolerance` times its largest value in the component.
    double tolerance = 1e-10;

    /// Most sweeps over a flow loop before giving up.
    int maxIterations = 1000;
};

struct UpwindTransportStatistics
{
    /// Number of components, i.e. cells or flow loops.
    int numComponents = 0;

    /// Number of components with more than one cell, i.e. flow loops.
    int numLoops = 0;

    /// Number of cells in the largest component.
    int largestComponent = 0;

    /// Most sweeps used for any flow loop that was solved by iteration.
    int maxIterations = 0;
};

namespace detail
{

    /// Solve the dense system `a x = b` with `a` of size m x m and `b` of size
    /// m x numRhs, both row-major. The solution replaces `b`, and `a` is destroyed.
    template <class Scalar>
    void solveDenseSystem(std::vector<Scalar>& a,
                          std::vector<Scalar>& b,
                          std::size_t m,
                          std::size_t numRhs)
    {
        for (std::size_t col = 0; col < m; ++col) {
            // Gaussian elimination with partial pivoting.
            std::size_t pivot = col;
            for (std::size_t row = col + 1; row < m; ++row) {
                if (std::abs(a[row * m + col]) > std::abs(a[pivot * m + col])) {
                    pivot = row;
                }
            }
            if (a[pivot * m + col] == Scalar {0}) {
                throw std::runtime_error("Opm::solveReordered: singular matrix");
            }
            if (pivot != col) {
                for (std::size_t k = 0; k < m; ++k) {
                    std::swap(a[pivot * m + k], a[col * m + k]);
                }
                for (std::size_t k = 0; k < numRhs; ++k) {
                    std::swap(b[pivot * numRhs + k], b[col * numRhs + k]);
                }
            }
            for (std::size_t row = col + 1; row < m; ++row) {
                const Scalar factor = a[row * m + col] / a[col * m + col];
                if (factor == Scalar {0}) {
                    continue;
                }
                for (std::size_t k = col; k < m; ++k) {
                    a[row * m + k] -= factor * a[col * m + k];
                }
                for (std::size_t k = 0; k < numRhs; ++k) {
                    b[row * numRhs + k] -= factor * b[col * numRhs + k];
                }
            }
        }
        for (std::size_t col = m; col-- > 0;) {
            for (std::size_t k = 0; k < numRhs; ++k) {
                Scalar sum = b[col * numRhs + k];
                for (std::size_t j = col + 1; j < m; ++j) {
                    sum -= a[col * m + j] * b[j * numRhs + k];
                }
                b[col * numRhs + k] = sum / a[col * m + col];
            }
        }
    }

    template <class Block>
    bool isNonzeroBlock(const Block& block)
    {
        for (std::size_t a = 0; a < Block::rows; ++a) {
            for (std::size_t b = 0; b < Block::cols; ++b) {
                if (block[a][b] != 0) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Call `f(row, column, block)` for every nonzero off-diagonal block of
    /// the matrix, row by row in the order of the matrix.
    template <class Matrix, class Function>
    void forEachNonzeroCoupling(const Matrix& matrix, Function&& f)
    {
        for (auto row = matrix.begin(); row != matrix.end(); ++row) {
            for (auto entry = row->begin(); entry != row->end(); ++entry) {
                if (entry.index() != row.index() && isNonzeroBlock(*entry)) {
                    f(row.index(), entry.index(), *entry);
                }
            }
        }
    }

    /// The coefficients of solveReordered(), from a sparse block matrix. Only
    /// the nonzero off-diagonal blocks couple cells.
    ///
    /// The interface that ReorderedSystem uses:
    ///  - `Scalar` and `blockSize`, the size N of the blocks.
    ///  - `numCells()`, and `graph()` with the cells each cell depends on.
    ///  - `diagonal(c)`, the N x N diagonal block.
    ///  - `forEachCoupling(c, f)`, which calls `f(u, block)` for each cell `u`
    ///    in row `c` of `graph()` with the off-diagonal block of the equation.
    template <class Matrix>
    class MatrixCoefficients
    {
        using Block = typename Matrix::block_type;
        static_assert(Block::rows == Block::cols, "The matrix blocks must be square");

    public:
        using Scalar = typename Block::field_type;
        static constexpr std::size_t blockSize = Block::rows;

        explicit MatrixCoefficients(const Matrix& matrix)
        {
            if (matrix.N() != matrix.M()) {
                throw std::invalid_argument("Opm::solveReordered: the matrix must be square");
            }
            diagonal_.reserve(matrix.N());
            for (std::size_t r = 0; r < matrix.N(); ++r) {
                const auto entry = matrix[r].find(r);
                if (entry == matrix[r].end()) {
                    throw std::invalid_argument("Opm::solveReordered: row " + std::to_string(r)
                                                + " has no diagonal block");
                }
                diagonal_.push_back(&*entry);
            }

            std::vector<int> rowStart(matrix.N() + 1, 0);
            std::vector<int> columns;
            std::vector<const Block*> blocks;
            forEachNonzeroCoupling(
                matrix, [&](std::size_t r, std::size_t c, const Block& block) {
                    ++rowStart[r + 1];
                    columns.push_back(static_cast<int>(c));
                    blocks.push_back(&block);
                });
            for (std::size_t r = 0; r < matrix.N(); ++r) {
                rowStart[r + 1] += rowStart[r];
            }
            graph_ = SparseTable<int>(std::move(columns), std::vector<int>(rowStart));
            couplings_ = SparseTable<const Block*>(std::move(blocks), std::move(rowStart));
        }

        std::size_t numCells() const
        {
            return diagonal_.size();
        }

        /// The dependency graph, as input to Opm::tarjan(). Row `r` lists the
        /// columns `c != r` with a nonzero block `A[r][c]`, in the order of the
        /// matrix row. Blocks that are stored but zero give no edge.
        const SparseTable<int>& graph() const
        {
            return graph_;
        }

        const Block& diagonal(std::size_t c) const
        {
            return *diagonal_[c];
        }

        template <class Function>
        void forEachCoupling(std::size_t c, Function&& f) const
        {
            auto block = couplings_[static_cast<int>(c)].begin();
            for (const int u : graph_[static_cast<int>(c)]) {
                f(static_cast<std::size_t>(u), **block++);
            }
        }

    private:
        std::vector<const Block*> diagonal_;
        SparseTable<int> graph_;
        SparseTable<const Block*> couplings_;
    };


    /// The system and the scratch space for solving it, with the coefficients
    /// of MatrixCoefficients.
    template <class Coefficients, class Vectors>
    class ReorderedSystem
    {
        using Scalar = typename Coefficients::Scalar;
        static constexpr std::size_t N = Coefficients::blockSize;

    public:
        ReorderedSystem(const Coefficients& coefficients,
                        const Vectors& rhs,
                        Vectors& x,
                        const UpwindTransportSettings& settings,
                        const std::string& name)
            : coefficients_(coefficients)
            , rhs_(rhs)
            , x_(x)
            , settings_(settings)
            , name_(name)
            , numCells_(coefficients.numCells())
            , numVectors_(rhs.size())
            , localIndex_(numCells_, -1)
            , cellSolution_(N * numVectors_)
            , change_(numVectors_)
            , scale_(numVectors_)
        {
            validate_();
        }

        UpwindTransportStatistics solve()
        {
            const auto components = tarjan(coefficients_.graph());
            UpwindTransportStatistics statistics;
            statistics.numComponents = components.numComponents();
            for (const auto cells : components.components()) {
                const int size = static_cast<int>(cells.size());
                statistics.largestComponent = std::max(statistics.largestComponent, size);
                if (size == 1) {
                    solveCell_(cells[0]);
                } else {
                    ++statistics.numLoops;
                    if (size <= settings_.maxDirectSize) {
                        solveDirectly_(cells);
                    } else {
                        const int sweeps = solveByGaussSeidel_(cells);
                        statistics.maxIterations = std::max(statistics.maxIterations, sweeps);
                    }
                }
            }
            return statistics;
        }

    private:
        void validate_() const
        {
            if (x_.size() != numVectors_) {
                throw std::invalid_argument(name_ + ": sizes do not match");
            }
            for (std::size_t q = 0; q < numVectors_; ++q) {
                if (static_cast<std::size_t>(rhs_[q].size()) != numCells_
                    || static_cast<std::size_t>(x_[q].size()) != numCells_) {
                    throw std::invalid_argument(name_ + ": wrong vector length");
                }
            }
        }

        /// The right-hand side of cell c for quantity q: `rhs` minus the
        /// couplings times the values of the cells that c depends on. The
        /// cells of a flow loop that is being solved directly are marked in
        /// `localIndex_`. They are unknowns of the dense system, so they are
        /// left out here. Everywhere else `localIndex_` is -1, and all
        /// couplings contribute.
        std::array<Scalar, N> rightHandSide_(std::size_t q, std::size_t c) const
        {
            std::array<Scalar, N> sum;
            for (std::size_t a = 0; a < N; ++a) {
                sum[a] = rhs_[q][c][a];
            }
            coefficients_.forEachCoupling(c, [&](std::size_t u, const auto& block) {
                if (localIndex_[u] < 0) {
                    for (std::size_t a = 0; a < N; ++a) {
                        for (std::size_t b = 0; b < N; ++b) {
                            sum[a] -= block[a][b] * x_[q][u][b];
                        }
                    }
                }
            });
            return sum;
        }

        /// Solve with the diagonal block of cell c for all quantities at once.
        /// The result is left in `cellSolution_`, N rows by numVectors columns.
        void solveDiagonal_(std::size_t c)
        {
            if constexpr (N == 1) {
                // One division
                const Scalar diagonal = coefficients_.diagonal(c)[0][0];
                if (diagonal == Scalar {0}) {
                    throw std::runtime_error(name_ + ": singular diagonal in row "
                                             + std::to_string(c));
                }
                for (std::size_t q = 0; q < numVectors_; ++q) {
                    cellSolution_[q] = rightHandSide_(q, c)[0] / diagonal;
                }
            } else {
                matrix_.resize(N * N);
                const auto& diagonal = coefficients_.diagonal(c);
                for (std::size_t a = 0; a < N; ++a) {
                    for (std::size_t b = 0; b < N; ++b) {
                        matrix_[a * N + b] = diagonal[a][b];
                    }
                }
                for (std::size_t q = 0; q < numVectors_; ++q) {
                    const auto sum = rightHandSide_(q, c);
                    for (std::size_t a = 0; a < N; ++a) {
                        cellSolution_[a * numVectors_ + q] = sum[a];
                    }
                }
                try {
                    solveDenseSystem(matrix_, cellSolution_, N, numVectors_);
                } catch (const std::runtime_error&) {
                    throw std::runtime_error(name_ + ": singular diagonal block in row "
                                             + std::to_string(c));
                }
            }
        }

        /// A cell that is not in a flow loop: one solve with its diagonal.
        void solveCell_(std::size_t c)
        {
            solveDiagonal_(c);
            for (std::size_t q = 0; q < numVectors_; ++q) {
                for (std::size_t a = 0; a < N; ++a) {
                    x_[q][c][a] = cellSolution_[a * numVectors_ + q];
                }
            }
        }

        /// A flow loop with a few cells: solve for all of its cells at once.
        void solveDirectly_(std::span<const int> cells)
        {
            const std::size_t m = cells.size();
            const std::size_t size = m * N;
            matrix_.assign(size * size, Scalar {0});
            solution_.assign(size * numVectors_, Scalar {0});
            for (std::size_t i = 0; i < m; ++i) {
                localIndex_[cells[i]] = static_cast<int>(i);
            }
            for (std::size_t i = 0; i < m; ++i) {
                const auto c = static_cast<std::size_t>(cells[i]);
                const auto& diagonal = coefficients_.diagonal(c);
                for (std::size_t a = 0; a < N; ++a) {
                    for (std::size_t b = 0; b < N; ++b) {
                        matrix_[(i * N + a) * size + i * N + b] = diagonal[a][b];
                    }
                }
                coefficients_.forEachCoupling(c, [&](std::size_t u, const auto& block) {
                    if (localIndex_[u] >= 0) {
                        const auto j = static_cast<std::size_t>(localIndex_[u]);
                        for (std::size_t a = 0; a < N; ++a) {
                            for (std::size_t b = 0; b < N; ++b) {
                                matrix_[(i * N + a) * size + j * N + b] += block[a][b];
                            }
                        }
                    }
                });
                for (std::size_t q = 0; q < numVectors_; ++q) {
                    // Cells outside the loop are already solved.
                    const auto sum = rightHandSide_(q, c);
                    for (std::size_t a = 0; a < N; ++a) {
                        solution_[(i * N + a) * numVectors_ + q] = sum[a];
                    }
                }
            }
            for (std::size_t i = 0; i < m; ++i) {
                localIndex_[cells[i]] = -1;
            }
            try {
                solveDenseSystem(matrix_, solution_, size, numVectors_);
            } catch (const std::runtime_error&) {
                throw std::runtime_error(name_ + ": singular matrix for a flow loop of "
                                         + std::to_string(m) + " cells");
            }
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t q = 0; q < numVectors_; ++q) {
                    for (std::size_t a = 0; a < N; ++a) {
                        x_[q][static_cast<std::size_t>(cells[i])][a]
                            = solution_[(i * N + a) * numVectors_ + q];
                    }
                }
            }
        }

        /// A large flow loop: Gauss-Seidel sweeps, in the order of the cells.
        /// Returns the number of sweeps.
        int solveByGaussSeidel_(std::span<const int> cells)
        {
            for (int sweep = 1; sweep <= settings_.maxIterations; ++sweep) {
                std::fill(change_.begin(), change_.end(), Scalar {0});
                std::fill(scale_.begin(), scale_.end(), Scalar {0});
                for (const int cell : cells) {
                    const auto c = static_cast<std::size_t>(cell);
                    solveDiagonal_(c);
                    for (std::size_t q = 0; q < numVectors_; ++q) {
                        for (std::size_t a = 0; a < N; ++a) {
                            auto& value = x_[q][c][a];
                            const Scalar updated = cellSolution_[a * numVectors_ + q];
                            change_[q] = std::max(change_[q], std::abs(updated - value));
                            scale_[q] = std::max(scale_[q], std::abs(updated));
                            value = updated;
                        }
                    }
                }
                bool converged = true;
                for (std::size_t q = 0; q < numVectors_; ++q) {
                    converged = converged && change_[q] <= settings_.tolerance * scale_[q];
                }
                if (converged) {
                    return sweep;
                }
            }
            throw std::runtime_error(name_ + ": no convergence after "
                                     + std::to_string(settings_.maxIterations)
                                     + " sweeps for a flow loop of " + std::to_string(cells.size())
                                     + " cells");
        }

        const Coefficients& coefficients_;
        const Vectors& rhs_;
        Vectors& x_;
        const UpwindTransportSettings& settings_;
        std::string name_;
        std::size_t numCells_;
        std::size_t numVectors_;

        // Scratch space, reused for every cell and flow loop
        std::vector<int> localIndex_;
        std::vector<Scalar> matrix_;
        std::vector<Scalar> solution_;
        std::vector<Scalar> cellSolution_;
        std::vector<Scalar> change_;
        std::vector<Scalar> scale_;
    };

} // namespace detail

/// Solve the block system `A x_q = b_q` described in the file comment.
///
/// \param matrix   Square sparse matrix with square N x N blocks, such as a
///                 `Dune::BCRSMatrix`. Every row must have a diagonal block.
/// \param rhs      One block vector of length `matrix.N()` per right-hand side.
/// \param x        The solution, in the layout of `rhs`. The values on entry
///                 are the initial guess when a flow loop is solved by
///                 iteration; otherwise they are not used.
/// \param settings When flow loops are solved directly, and the tolerance and
///                 iteration limit of the Gauss-Seidel sweeps otherwise.
///
/// \return Statistics about the components that were found and solved.
///
/// \throws std::invalid_argument if the matrix is not square, the sizes do not
///         match, or a row has no diagonal block.
/// \throws std::runtime_error if a diagonal block or the matrix of a flow loop
///         is singular, or if the iteration for a flow loop does not converge.
template <class Matrix, class Vectors>
UpwindTransportStatistics
solveReordered(const Matrix& matrix,
               const Vectors& rhs,
               Vectors& x,
               const UpwindTransportSettings& settings = {})
{
    const detail::MatrixCoefficients<Matrix> coefficients(matrix);
    return detail::ReorderedSystem<detail::MatrixCoefficients<Matrix>, Vectors>(
               coefficients, rhs, x, settings, "Opm::solveReordered")
        .solve();
}

} // namespace Opm

#endif // OPM_UPWIND_TRANSPORT_SOLVER_HEADER_INCLUDED
