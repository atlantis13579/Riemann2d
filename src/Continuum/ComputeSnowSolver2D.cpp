#include "Continuum/SnowGpuSolver2D.h"
#include "Fluid/Compute/ComputeDevice.h"
#include "SnowComputeShaders.h"

#include <stdexcept>

namespace Riemann2D
{
    namespace
    {
        using namespace Compute;
        class ComputeSnowSolver final : public SnowGpuSolver2D
        {
            std::unique_ptr<Device> Queue;
            SharedBuffer Particles, Grid, Heads, Next, Stress;
            SharedProgram Clear, Prepare, TransferGrid, TransferParticles;
            std::vector<SnowParticle2D> Readback;
        public:
            bool Step(const SnowSettings2D& settings, const SnowCapsule2D& capsule, float dt,
                      std::vector<SnowParticle2D>& particles, std::string& error) override
            {
                try
                {
                    if (!Queue)
                    {
                        Queue = CreateDevice();
                        if (!Queue) throw std::runtime_error("Native compute support was not built");
                        Clear = Queue->CreateProgram(SnowShaders, "SnowClear");
                        Prepare = Queue->CreateProgram(SnowShaders, "SnowPrepare");
                        TransferGrid = Queue->CreateProgram(SnowShaders, "SnowGrid");
                        TransferParticles = Queue->CreateProgram(SnowShaders, "SnowParticles");
                    }
                    if (particles.empty()) return true;
                    const int count = static_cast<int>(particles.size());
                    const int width = static_cast<int>(std::ceil((settings.BoundsMaximum.X - settings.BoundsMinimum.X) / settings.CellSize)) + 1;
                    const int height = static_cast<int>(std::ceil((settings.BoundsMaximum.Y - settings.BoundsMinimum.Y) / settings.CellSize)) + 1;
                    const int cells = width * height;
                    Constants p;
                    p.Integer(0,count); p.Integer(1,width); p.Integer(2,height); p.Integer(3,static_cast<int>(settings.Transfer));
                    p.Float(4,settings.BoundsMinimum.X); p.Float(5,settings.BoundsMinimum.Y);
                    p.Float(6,settings.BoundsMaximum.X); p.Float(7,settings.BoundsMaximum.Y);
                    p.Float(8,settings.CellSize); p.Float(9,dt); p.Float(10,settings.Gravity.X); p.Float(11,settings.Gravity.Y);
                    const auto& m = settings.Material;
                    p.Float(12,m.YoungModulus); p.Float(13,m.PoissonRatio); p.Float(14,m.SnowFraction); p.Float(15,m.Viscosity);
                    p.Float(16,m.Hardening); p.Float(17,m.CompressionYield); p.Float(18,m.StretchYield); p.Float(19,m.FrictionAngle);
                    p.Float(20,settings.FlipRatio); p.Float(21,capsule.Center.X); p.Float(22,capsule.Center.Y);
                    p.Float(23,capsule.Velocity.X); p.Float(24,capsule.Velocity.Y); p.Float(25,capsule.HalfLength);
                    p.Float(26,capsule.Radius); p.Float(27,capsule.Friction); p.Integer(28,capsule.Enabled ? 1 : 0);
                    p.Float(29,settings.FixedBedHeight);
                    static_assert(sizeof(SnowParticle2D) == 16 * sizeof(float), "MPM shader layout mismatch");
                    Reserve(*Queue, Particles, count * sizeof(SnowParticle2D));
                    Reserve(*Queue, Grid, cells * 6 * sizeof(float));
                    Reserve(*Queue, Heads, cells * sizeof(int));
                    Reserve(*Queue, Next, count * sizeof(int));
                    Reserve(*Queue, Stress, count * 4 * sizeof(float));
                    Readback.resize(count);
                    Queue->Upload(Particles, particles.data(), count * sizeof(SnowParticle2D));
                    const Bindings bindings{Particles,Grid,Heads,Next,Stress};
                    Queue->Dispatch(Clear,bindings,p,cells);
                    Queue->Dispatch(Prepare,bindings,p,count);
                    Queue->Dispatch(TransferGrid,bindings,p,cells);
                    Queue->Dispatch(TransferParticles,bindings,p,count);
                    Queue->Download({{Particles,Readback.data(),count * sizeof(SnowParticle2D)}});
                    for (const auto& particle : Readback)
                        if (!std::isfinite(particle.Position.X + particle.Position.Y + particle.Velocity.X + particle.Velocity.Y + particle.Elastic.Determinant() + particle.PlasticVolume))
                            throw std::runtime_error("Compute MPM produced a non-finite state");
                    particles.swap(Readback); error.clear(); return true;
                }
                catch (const std::exception& exception)
                {
                    if (Queue) Queue->Finish();
                    error = exception.what(); return false;
                }
            }
        };
    }

    std::unique_ptr<SnowGpuSolver2D> CreateComputeSnowSolver2D() { return std::make_unique<ComputeSnowSolver>(); }
}
