#include "LCPSolver2D.h"

#include "Dynamics/ContactSolverUtilities2D.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace Riemann2D
{
    namespace
    {
        struct ConstraintRow
        {
            Contact2* Contact = nullptr;
            Vector2 RadiusA;
            Vector2 RadiusB;
            float RightHandSide = 0.0f;
        };

        Vector2 LinearJacobian(const ConstraintRow& row, const RigidBody2D* body)
        {
            if (body == row.Contact->BodyA) return -row.Contact->Normal;
            if (body == row.Contact->BodyB) return row.Contact->Normal;
            return Vector2::Zero();
        }

        float AngularJacobian(const ConstraintRow& row, const RigidBody2D* body)
        {
            if (body == row.Contact->BodyA) return -row.RadiusA.Cross(row.Contact->Normal);
            if (body == row.Contact->BodyB) return row.RadiusB.Cross(row.Contact->Normal);
            return 0.0f;
        }

        float Coupling(const ConstraintRow& first, const ConstraintRow& second)
        {
            float value = 0.0f;
            const RigidBody2D* bodies[] = {first.Contact->BodyA, first.Contact->BodyB};
            for (const RigidBody2D* body : bodies)
            {
                const Vector2 linearA = LinearJacobian(first, body);
                const Vector2 linearB = LinearJacobian(second, body);
                const float angularA = AngularJacobian(first, body);
                const float angularB = AngularJacobian(second, body);
                value += body->GetInverseMass() * linearA.Dot(linearB) +
                         body->GetInverseInertia() * angularA * angularB;
            }
            return value;
        }

        bool SolveLinearSystem(std::vector<float> matrix, std::vector<float> rhs,
                               std::vector<float>& solution, int size)
        {
            solution.assign(static_cast<std::size_t>(size), 0.0f);
            for (int column = 0; column < size; ++column)
            {
                int pivot = column;
                float pivotMagnitude = std::abs(matrix[static_cast<std::size_t>(column * size + column)]);
                for (int row = column + 1; row < size; ++row)
                {
                    const float magnitude = std::abs(matrix[static_cast<std::size_t>(row * size + column)]);
                    if (magnitude > pivotMagnitude)
                    {
                        pivot = row;
                        pivotMagnitude = magnitude;
                    }
                }
                if (pivotMagnitude < 1.0e-7f)
                {
                    return false;
                }
                if (pivot != column)
                {
                    for (int entry = column; entry < size; ++entry)
                    {
                        std::swap(matrix[static_cast<std::size_t>(column * size + entry)],
                                  matrix[static_cast<std::size_t>(pivot * size + entry)]);
                    }
                    std::swap(rhs[static_cast<std::size_t>(column)], rhs[static_cast<std::size_t>(pivot)]);
                }
                const float diagonal = matrix[static_cast<std::size_t>(column * size + column)];
                for (int row = column + 1; row < size; ++row)
                {
                    const float factor = matrix[static_cast<std::size_t>(row * size + column)] / diagonal;
                    for (int entry = column; entry < size; ++entry)
                    {
                        matrix[static_cast<std::size_t>(row * size + entry)] -=
                            factor * matrix[static_cast<std::size_t>(column * size + entry)];
                    }
                    rhs[static_cast<std::size_t>(row)] -= factor * rhs[static_cast<std::size_t>(column)];
                }
            }
            for (int row = size - 1; row >= 0; --row)
            {
                float value = rhs[static_cast<std::size_t>(row)];
                for (int column = row + 1; column < size; ++column)
                {
                    value -= matrix[static_cast<std::size_t>(row * size + column)] *
                             solution[static_cast<std::size_t>(column)];
                }
                solution[static_cast<std::size_t>(row)] = value /
                    matrix[static_cast<std::size_t>(row * size + row)];
            }
            return true;
        }

        void Multiply(const std::vector<float>& matrix, const std::vector<float>& vector,
                      std::vector<float>& result, int size, SolverBackend2D backend)
        {
            DenseMatrixVectorMultiply(matrix.data(), vector.data(), result.data(),
                                      size, backend);
        }

        void SolveActiveSet(const std::vector<float>& matrix, const std::vector<float>& q,
                            std::vector<float>& lambda, std::vector<float>& residual,
                            int size, SolverBackend2D backend)
        {
            lambda.assign(static_cast<std::size_t>(size), 0.0f);
            residual = q;
            std::vector<bool> active(static_cast<std::size_t>(size), false);

            for (int iteration = 0; iteration < size * 6 + 8; ++iteration)
            {
                int addIndex = -1;
                float mostNegative = -1.0e-5f;
                for (int index = 0; index < size; ++index)
                {
                    if (!active[static_cast<std::size_t>(index)] &&
                        residual[static_cast<std::size_t>(index)] < mostNegative)
                    {
                        mostNegative = residual[static_cast<std::size_t>(index)];
                        addIndex = index;
                    }
                }
                if (addIndex >= 0)
                {
                    active[static_cast<std::size_t>(addIndex)] = true;
                }

                std::vector<int> activeIndices;
                for (int index = 0; index < size; ++index)
                {
                    if (active[static_cast<std::size_t>(index)]) activeIndices.push_back(index);
                }
                if (activeIndices.empty())
                {
                    break;
                }

                const int activeSize = static_cast<int>(activeIndices.size());
                std::vector<float> activeMatrix(static_cast<std::size_t>(activeSize * activeSize));
                std::vector<float> activeRhs(static_cast<std::size_t>(activeSize));
                for (int row = 0; row < activeSize; ++row)
                {
                    activeRhs[static_cast<std::size_t>(row)] =
                        -q[static_cast<std::size_t>(activeIndices[static_cast<std::size_t>(row)])];
                    for (int column = 0; column < activeSize; ++column)
                    {
                        activeMatrix[static_cast<std::size_t>(row * activeSize + column)] = matrix[
                            static_cast<std::size_t>(activeIndices[static_cast<std::size_t>(row)] * size +
                                                     activeIndices[static_cast<std::size_t>(column)])];
                    }
                }
                std::vector<float> activeSolution;
                if (!SolveLinearSystem(activeMatrix, activeRhs, activeSolution, activeSize))
                {
                    active[static_cast<std::size_t>(activeIndices.back())] = false;
                    continue;
                }

                std::fill(lambda.begin(), lambda.end(), 0.0f);
                int removeIndex = -1;
                float mostNegativeLambda = -1.0e-5f;
                for (int index = 0; index < activeSize; ++index)
                {
                    const int originalIndex = activeIndices[static_cast<std::size_t>(index)];
                    const float value = activeSolution[static_cast<std::size_t>(index)];
                    lambda[static_cast<std::size_t>(originalIndex)] = value;
                    if (value < mostNegativeLambda)
                    {
                        mostNegativeLambda = value;
                        removeIndex = originalIndex;
                    }
                }
                if (removeIndex >= 0)
                {
                    active[static_cast<std::size_t>(removeIndex)] = false;
                    continue;
                }

                Multiply(matrix, lambda, residual, size, backend);
                for (int index = 0; index < size; ++index)
                {
                    residual[static_cast<std::size_t>(index)] += q[static_cast<std::size_t>(index)];
                }

                bool feasible = true;
                for (int index = 0; index < size; ++index)
                {
                    if (!active[static_cast<std::size_t>(index)] &&
                        residual[static_cast<std::size_t>(index)] < -1.0e-5f)
                    {
                        feasible = false;
                        break;
                    }
                }
                if (feasible)
                {
                    break;
                }
            }
        }
    }

    void LCPSolver2D::Solve(std::vector<Contact2>& contacts, SolverBackend2D backend)
    {
        LastResidual = 0.0f;
        if (contacts.empty())
        {
            return;
        }

        const int size = static_cast<int>(contacts.size());
        std::vector<ConstraintRow> rows(static_cast<std::size_t>(size));
        std::vector<float> q(static_cast<std::size_t>(size));
        std::vector<float> matrix(static_cast<std::size_t>(size * size));
        for (int index = 0; index < size; ++index)
        {
            Contact2& contact = contacts[static_cast<std::size_t>(index)];
            contact.NormalImpulse = 0.0f;
            contact.TangentImpulse = 0.0f;
            ConstraintRow& row = rows[static_cast<std::size_t>(index)];
            row.Contact = &contact;
            row.RadiusA = contact.Point - contact.BodyA->GetTransform().Position;
            row.RadiusB = contact.Point - contact.BodyB->GetTransform().Position;
            const Vector2 relativeVelocity =
                SolverDetail::VelocityAtPoint(*contact.BodyB, contact.Point) -
                SolverDetail::VelocityAtPoint(*contact.BodyA, contact.Point);
            const float normalVelocity = relativeVelocity.Dot(contact.Normal);
            const float restitution = normalVelocity < -1.0f
                ? std::max(contact.BodyA->GetRestitution(), contact.BodyB->GetRestitution()) : 0.0f;
            q[static_cast<std::size_t>(index)] = (1.0f + restitution) * normalVelocity;
        }
        for (int row = 0; row < size; ++row)
        {
            for (int column = 0; column < size; ++column)
            {
                matrix[static_cast<std::size_t>(row * size + column)] =
                    Coupling(rows[static_cast<std::size_t>(row)], rows[static_cast<std::size_t>(column)]);
            }
        }

        std::vector<float> lambda;
        std::vector<float> residual(static_cast<std::size_t>(size));
        SolveActiveSet(matrix, q, lambda, residual, size, backend);
        for (int index = 0; index < size; ++index)
        {
            Contact2& contact = contacts[static_cast<std::size_t>(index)];
            contact.NormalImpulse = std::max(lambda[static_cast<std::size_t>(index)], 0.0f);
            SolverDetail::ApplyImpulse(contact, contact.NormalImpulse * contact.Normal);
            LastResidual = std::max(LastResidual,
                std::max(std::max(-lambda[static_cast<std::size_t>(index)],
                                  -residual[static_cast<std::size_t>(index)]),
                         std::abs(lambda[static_cast<std::size_t>(index)] *
                                  residual[static_cast<std::size_t>(index)])));
        }

        for (int iteration = 0; iteration < 4; ++iteration)
        {
            for (Contact2& contact : contacts)
            {
                const Vector2 tangent = SolverDetail::Tangent(contact.Normal);
                const Vector2 relativeVelocity =
                    SolverDetail::VelocityAtPoint(*contact.BodyB, contact.Point) -
                    SolverDetail::VelocityAtPoint(*contact.BodyA, contact.Point);
                const float maximumFriction = SolverDetail::MixedFriction(contact) * contact.NormalImpulse;
                const float oldImpulse = contact.TangentImpulse;
                contact.TangentImpulse = Clamp(oldImpulse -
                    SolverDetail::EffectiveMass(contact, tangent) * relativeVelocity.Dot(tangent),
                    -maximumFriction, maximumFriction);
                SolverDetail::ApplyImpulse(contact, (contact.TangentImpulse - oldImpulse) * tangent);
            }
        }
    }
}
