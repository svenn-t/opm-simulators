/*
  Copyright 2026 NORCE.

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
#ifndef ORTHOMIN_HPP
#define ORTHOMIN_HPP

#include <dune/istl/solver.hh>

#include <vector>

namespace Dune
{

template <class X>
class Orthomin : public Dune::IterativeSolver<X, X>
{
public:
    using typename IterativeSolver<X, X>::domain_type;
    using typename IterativeSolver<X, X>::range_type;
    using typename IterativeSolver<X, X>::field_type;
    using typename IterativeSolver<X, X>::real_type;

    // copy base class constructors
    using IterativeSolver<X, X>::IterativeSolver;

    // don't shadow four-argument version of apply defined in the base class
    using IterativeSolver<X, X>::apply;

    // Helper struct for storing NSTACK vectors
    template <typename T>
    struct NStackVector
    {
        std::vector<T> vec;
        std::size_t head{0};
        std::size_t capacity;
        bool filled{false};

        NStackVector(std::size_t cap) : vec(cap), capacity(cap)
        {
        }

        void push(const T& x)
        {
            // Insert in vec at head by copy assignment
            vec[head] = x;

            // Update head and check if filled
            head = (head + 1) % capacity;
            if (head == 0) {
                filled = true;
            }
        }

        // template <typename... Args>
        // void emplace(Args&&... args)
        // {
        //     // Insert vec at head by construction in place
        //     vec[head] = T(std::forward<Args>(args)...);
        //
        //     // Update head and check if filled
        //     head = (head + 1) % capacity;
        //     if (head == 0) {
        //         filled = true;
        //     }
        // }

        T& operator[](std::size_t i)
        {
            // If vec is filled, we need the correct starting index
            std::size_t start = filled ? head : 0;
            std::size_t ind = (start + i) % capacity;
            return vec[ind];
        }

        const T& operator[](std::size_t i) const
        {
            // If vec is filled, we need the correct starting index
            std::size_t start = filled ? head : 0;
            std::size_t ind = (start + i) % capacity;
            return vec[ind];
        }

        std::size_t size() const
        {
            // return max(head, nstack)
            return filled ? capacity : head;
        }
    };

    /*!
     *
     * @param x Result vector (init. guess on input)
     * @param b Defect vector (right-hand side vector on input)
     * @param res Results statistics
     *
     * Solves a preconditioned linear system using the ORTHOMIN algorithm from Vinsome 1976.
     */
    void apply(X& x, X& b, InverseOperatorResult& res) override
    {
        // Max. no. of search vectors to use
        const std::size_t nstack = 10;

        // Init. iteration helper
        Iteration iteration(*this, res);

        // Prepare preconditioner and calc. init. defect (r = x - Ab)
        _prec->pre(x, b);
        _op->applyscaleadd(-1, x, b); // OBS: overwrites b

        // Check convergence before iterations starts
        real_type norm = _sp->norm(b);
        if (iteration.step(0, norm)) {
            // Post-process preconditioner and return
            _prec->post(x);
            return;
        }

        // Init. scalar prod. storage (beta)
        NStackVector<real_type> beta(nstack);

        // Init. search vectors q and p (= A * q), and calculate q[0] and p[0]
        NStackVector<X> q(nstack);
        NStackVector<X> p(nstack);
        X xzero(x);
        xzero = 0.0;
        q.push(xzero);
        p.push(xzero);
        _prec->apply(q[0], b);
        _op->apply(q[0], p[0]);

        // Iteration loop
        unsigned k = 1;
        while (static_cast<int>(k) <= _maxit) {
            // New omega (for updating x and b)
            field_type sprod1 = _sp->dot(p[k - 1], b);
            field_type beta_k = _sp->dot(p[k - 1], p[k - 1]);
            beta.push(beta_k);

            field_type omega = Simd::cond(norm == field_type(0.0),
                                          field_type(0.0),
                                          field_type(sprod1 / beta_k));

            // Update x and defect
            x.axpy(omega, q[k - 1]);
            b.axpy(-omega, p[k - 1]);

            // Check convergence
            norm = _sp->norm(b);
            if (iteration.step(k, norm)) {
                _prec->post(x);
                return;
            }

            // Init. search vectors before update
            q.push(xzero);
            p.push(xzero);
            _prec->apply(q[k], b);
            _op->apply(q[k], p[k]);

            // Update search vectors and store last scalar product
            // OBS: We only use at most nstack search vectors, hence q.size() = max(k + 1, nstack)
            const auto p_k = p[k]; // need a copy for scalar product
            for (unsigned i = 0; i < q.size() - 1; ++i) {
                field_type sprod2 = _sp->dot(p_k, p[i]);
                if (i == q.size() - 2) {
                    field_type beta_i = _sp->dot(p[i], p[i]);
                    beta.push(beta_i);
                }
                field_type alpha_i = Simd::cond(norm == field_type(0.0),
                                                field_type(0.0),
                                                field_type(sprod2 / beta[i]));

                q[k].axpy(-alpha_i, q[i]);
                p[k].axpy(-alpha_i, p[i]);
            }
            ++k;
        }
    }

protected:
    using IterativeSolver<X, X>::_op;
    using IterativeSolver<X, X>::_prec;
    using IterativeSolver<X, X>::_sp;
    using IterativeSolver<X, X>::_reduction;
    using IterativeSolver<X, X>::_maxit;
    using IterativeSolver<X, X>::_verbose;
    using Iteration = typename IterativeSolver<X, X>::template Iteration<unsigned>;
};

}

#endif //ORTHOMIN_HPP
