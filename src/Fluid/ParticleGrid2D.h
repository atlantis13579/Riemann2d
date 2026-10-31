#pragma once

#include "Maths/Vector2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace Riemann2D
{
    // Compact counting-sort hash grid. Exact cell checks reject hash collisions.
    class ParticleGrid2D
    {
    public:
        template<class Particle>
        void Build(const std::vector<Particle>& particles, float cellSize)
        {
            CellSize = std::max(cellSize, 1.0e-4f);
            std::size_t capacity = 16;
            while (capacity < particles.size() * 2) capacity *= 2;
            Mask = capacity - 1;
            Offsets.assign(capacity + 1, 0);
            Cells.resize(particles.size());
            Indices.resize(particles.size());
            for (std::size_t i = 0; i < particles.size(); ++i)
            {
                Cells[i] = Cell(particles[i].Position);
                ++Offsets[Hash(Cells[i].X, Cells[i].Y) + 1];
            }
            for (std::size_t i = 1; i < Offsets.size(); ++i) Offsets[i] += Offsets[i - 1];
            Cursors = Offsets;
            for (std::size_t i = 0; i < particles.size(); ++i)
                Indices[Cursors[Hash(Cells[i].X, Cells[i].Y)]++] = i;
        }

        template<class Visitor>
        void Query(const Vector2& position, float radius, Visitor visitor) const
        {
            if (Indices.empty()) return;
            const Coordinate low = Cell(position - Vector2{radius, radius});
            const Coordinate high = Cell(position + Vector2{radius, radius});
            for (int y = low.Y; y <= high.Y; ++y)
                for (int x = low.X; x <= high.X; ++x)
                {
                    const std::size_t bucket = Hash(x, y);
                    for (std::size_t entry = Offsets[bucket]; entry < Offsets[bucket + 1]; ++entry)
                    {
                        const std::size_t index = Indices[entry];
                        if (Cells[index].X == x && Cells[index].Y == y) visitor(index);
                    }
                }
        }

    private:
        struct Coordinate { int X; int Y; };
        Coordinate Cell(const Vector2& position) const
        {
            return {static_cast<int>(std::floor(position.X / CellSize)),
                    static_cast<int>(std::floor(position.Y / CellSize))};
        }
        std::size_t Hash(int x, int y) const
        {
            return ((static_cast<std::uint32_t>(x) * 73856093u) ^
                    (static_cast<std::uint32_t>(y) * 19349663u)) & Mask;
        }
        float CellSize = 1.0f;
        std::size_t Mask = 15;
        std::vector<Coordinate> Cells;
        std::vector<std::size_t> Offsets;
        std::vector<std::size_t> Cursors;
        std::vector<std::size_t> Indices;
    };
}
