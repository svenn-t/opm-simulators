// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2025, NORCE AS

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
#include <config.h>

#define BOOST_TEST_MODULE TpsaSystemTests

#include <boost/test/unit_test.hpp>

#include <opm/simulators/linalg/system/MatrixResidualSplitterTPSA.hpp>
#include <opm/simulators/linalg/system/SystemTypes.hpp>

using namespace Opm;

struct setupTest
{
    using Matrix = Dune::BCRSMatrix<MatrixBlock<double, 7, 7>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<double, 7>>;
    using Splitter = MatrixResidualSplitterTPSA<double, Matrix, Vector>;

    setupTest()
    {
        // Set matrix entries
        mat = Matrix(1, 1, Matrix::random);
        mat.setrowsize(0, 1);
        mat.endrowsizes();
        mat.addindex(0, 0);
        mat.endindices();
        auto& block = mat[0][0];
        for (std::size_t i = 0; i < 7; ++i) {
            for (std::size_t j = 0; j < 7; ++j) {
                block[i][j] = static_cast<double>(i) * 7 + static_cast<double>(j);
            }
        }

        // Set vector entries
        vec = Vector(1);
        auto& fvec = vec[0];
        for (std::size_t k = 0; k < 7; ++k) {
            fvec[k] = static_cast<double>(k);
        }

        // Split to 9 sub-matrices and 3 sub-vectors
        std::vector<std::set<unsigned>> sparsityPattern(1);
        sparsityPattern[0].insert(0);
        splitter = Splitter(mat, vec, sparsityPattern);
        splitter.generateSubmatrices();
        splitter.generateSubResiduals();
    }

    Matrix mat;
    Vector vec;
    Splitter splitter;
};

BOOST_AUTO_TEST_CASE(TestMatrixVectorSplit)
{
    using Matrix = Dune::BCRSMatrix<MatrixBlock<double, 7, 7>>;
    using Vector = Dune::BlockVector<Dune::FieldVector<double, 7>>;
    using Splitter = MatrixResidualSplitterTPSA<double, Matrix, Vector>;

    // Set matrix entries
    Matrix mat(1, 1, Matrix::random);
    mat.setrowsize(0, 1);
    mat.endrowsizes();
    mat.addindex(0, 0);
    mat.endindices();
    auto& block = mat[0][0];
    for (std::size_t i = 0; i < 7; ++i) {
        for (std::size_t j = 0; j < 7; ++j) {
            block[i][j] = static_cast<double>(i) * 7 + static_cast<double>(j);
        }
    }

    // Set vector entries
    Vector vec(1);
    auto& fvec = vec[0];
    for (std::size_t k = 0; k < 7; ++k) {
        fvec[k] = static_cast<double>(k);
    }

    // Split to 9 sub-matrices and 3 sub-vectors
    std::vector<std::set<unsigned>> sparsityPattern(1);
    sparsityPattern[0].insert(0);
    Splitter splitter(mat, vec, sparsityPattern);
    splitter.generateSubmatrices();
    splitter.generateSubResiduals();

    // Check and see if sub-matrices and sub-vectors are equal the original matrix and vector
    const std::unique_ptr<DispDispMatrixT<double>> mat00 = splitter.takeDispDispMatrix();
    const auto* mat00Block = mat00->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat00->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat00->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat00Block->N(), 3);
    BOOST_CHECK_EQUAL(mat00Block->M(), 3);
    for (std::size_t l = 0; l < mat00Block->N(); ++l) {
        for (std::size_t m = 0; m < mat00Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat00Block)[l][m], block[l][m]);
        }
    }

    const std::unique_ptr<DispRotMatrixT<double>> mat01 = splitter.takeDispRotMatrix();
    const auto* mat01Block = mat01->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat01->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat01->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat01Block->N(), 3);
    BOOST_CHECK_EQUAL(mat01Block->M(), 3);
    for (std::size_t l = 0; l < mat01Block->N(); ++l) {
        for (std::size_t m = 0; m < mat01Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat01Block)[l][m], block[l][m+3]);
        }
    }

    const std::unique_ptr<DispSPresMatrixT<double>> mat02 = splitter.takeDispSPresMatrix();
    const auto* mat02Block = mat02->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat02->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat02->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat02Block->N(), 3);
    BOOST_CHECK_EQUAL(mat02Block->M(), 1);
    for (std::size_t l = 0; l < mat02Block->N(); ++l) {
        for (std::size_t m = 0; m < mat02Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat02Block)[l][m], block[l][m+6]);
        }
    }

    const std::unique_ptr<RotDispMatrixT<double>> mat10 = splitter.takeRotDispMatrix();
    const auto* mat10Block = mat10->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat10->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat10->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat10Block->N(), 3);
    BOOST_CHECK_EQUAL(mat10Block->M(), 3);
    for (std::size_t l = 0; l < mat10Block->N(); ++l) {
        for (std::size_t m = 0; m < mat10Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat10Block)[l][m], block[l+3][m]);
        }
    }

    const std::unique_ptr<RotRotMatrixT<double>> mat11 = splitter.takeRotRotMatrix();
    const auto* mat11Block = mat11->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat11->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat11->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat11Block->N(), 3);
    BOOST_CHECK_EQUAL(mat11Block->M(), 3);
    for (std::size_t l = 0; l < mat11Block->N(); ++l) {
        for (std::size_t m = 0; m < mat11Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat11Block)[l][m], block[l+3][m+3]);
        }
    }

    const std::unique_ptr<RotSPresMatrixT<double>> mat12 = splitter.takeRotSPresMatrix();
    const auto* mat12Block = mat12->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat12->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat12->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat12Block->N(), 3);
    BOOST_CHECK_EQUAL(mat12Block->M(), 1);
    for (std::size_t l = 0; l < mat12Block->N(); ++l) {
        for (std::size_t m = 0; m < mat12Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat12Block)[l][m], block[l+3][m+6]);
        }
    }

    const std::unique_ptr<SPresDispMatrixT<double>> mat20 = splitter.takeSPresDispMatrix();
    const auto* mat20Block = mat20->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat20->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat20->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat20Block->N(), 1);
    BOOST_CHECK_EQUAL(mat20Block->M(), 3);
    for (std::size_t l = 0; l < mat20Block->N(); ++l) {
        for (std::size_t m = 0; m < mat20Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat20Block)[l][m], block[l+6][m]);
        }
    }

    const std::unique_ptr<SPresRotMatrixT<double>> mat21 = splitter.takeSPresRotMatrix();
    const auto* mat21Block = mat21->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat21->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat21->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat21Block->N(), 1);
    BOOST_CHECK_EQUAL(mat21Block->M(), 3);
    for (std::size_t l = 0; l < mat21Block->N(); ++l) {
        for (std::size_t m = 0; m < mat21Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat21Block)[l][m], block[l+6][m+3]);
        }
    }

    const std::unique_ptr<SPresSPresMatrixT<double>> mat22 = splitter.takeSPresSPresMatrix();
    const auto* mat22Block = mat22->blockAddress(0,0);
    BOOST_CHECK_EQUAL(mat22->istlMatrix().N(), 1);
    BOOST_CHECK_EQUAL(mat22->istlMatrix().M(), 1);
    BOOST_CHECK_EQUAL(mat22Block->N(), 1);
    BOOST_CHECK_EQUAL(mat22Block->M(), 1);
    for (std::size_t l = 0; l < mat22Block->N(); ++l) {
        for (std::size_t m = 0; m < mat22Block->M(); ++m) {
            BOOST_CHECK_EQUAL((*mat22Block)[l][m], block[l+6][m+6]);
        }
    }

    const auto& vec0 = splitter.dispVector();
    BOOST_CHECK_EQUAL(vec0.size(), 1);
    BOOST_CHECK_EQUAL(vec0[0].size(), 3);
    for (std::size_t l = 0; l < vec0[0].size(); ++l) {
        BOOST_CHECK_EQUAL(vec0[0][l], fvec[l]);
    }

    const auto& vec1 = splitter.rotVector();
    BOOST_CHECK_EQUAL(vec1.size(), 1);
    BOOST_CHECK_EQUAL(vec1[0].size(), 3);
    for (std::size_t l = 0; l < vec1[0].size(); ++l) {
        BOOST_CHECK_EQUAL(vec1[0][l], fvec[l+3]);
    }

    const auto& vec2 = splitter.sPresVector();
    BOOST_CHECK_EQUAL(vec2.size(), 1);
    BOOST_CHECK_EQUAL(vec2[0].size(), 1);
    for (std::size_t l = 0; l < vec2[0].size(); ++l) {
        BOOST_CHECK_EQUAL(vec2[0][l], fvec[l+6]);
    }

    // System matrix and vector initialization
    SystemMatrixT<double> sysMat;
    SystemVectorT<double> sysVec;

    sysMat.M11 = mat00.get();
    sysMat.M12 = mat01.get();
    sysMat.M13 = mat02.get();
    sysMat.M21 = mat10.get();
    sysMat.M22 = mat11.get();
    sysMat.M23 = mat12.get();
    sysMat.M31 = mat20.get();
    sysMat.M32 = mat21.get();
    sysMat.M33 = mat22.get();

    using namespace Dune::Indices;
    sysVec[_0] = splitter.takeDispVector();
    sysVec[_1] = splitter.takeRotVector();
    sysVec[_2] = splitter.takeSPresVector();

    // Test system matrix operators
    SystemVectorT<double> resVec;
    resVec[_0].resize(1);
    resVec[_0] = 0.0;
    resVec[_1].resize(1);
    resVec[_1] = 0.0;
    resVec[_2].resize(1);
    resVec[_2] = 0.0;

    // Reference vector init
    Vector ref;
    ref.resize(1);
    ref[0] = 0.0;

    // mv
    mat.mv(vec, ref);
    sysMat.mv(sysVec, resVec);
    for (std::size_t i = 0; i < resVec[_0].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_0][0][i], ref[0][i]);
    }
    for (std::size_t i = 0; i < resVec[_1].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_1][0][i], ref[0][i+3]);
    }
    for (std::size_t i = 0; i < resVec[_1].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_2][0][i], ref[0][i+6]);
    }

    // umv
    mat.umv(vec, ref);
    sysMat.umv(sysVec, resVec);
    for (std::size_t i = 0; i < resVec[_0].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_0][0][i], ref[0][i]);
    }
    for (std::size_t i = 0; i < resVec[_1].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_1][0][i], ref[0][i+3]);
    }
    for (std::size_t i = 0; i < resVec[_1].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_2][0][i], ref[0][i+6]);
    }

    // usmv
    SystemMatrixT<double>::field_type alpha = 2.0;
    mat.usmv(alpha, vec, ref);
    sysMat.usmv(alpha, sysVec, resVec);
    for (std::size_t i = 0; i < resVec[_0].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_0][0][i], ref[0][i]);
    }
    for (std::size_t i = 0; i < resVec[_1].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_1][0][i], ref[0][i+3]);
    }
    for (std::size_t i = 0; i < resVec[_1].size(); ++i) {
        BOOST_CHECK_EQUAL(resVec[_2][0][i], ref[0][i+6]);
    }

}
