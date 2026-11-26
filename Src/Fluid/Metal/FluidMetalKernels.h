// Native Metal kernels.
R"SHADER(

#include <metal_stdlib>
using namespace metal;

float ReadFloat(device uint *b, uint i)
{
    return as_type<float>(b[i]);
}

uint ReadUInt(device uint *b, uint i)
{
    return b[i];
}

void WriteFloat(device uint *b, uint i, float v)
{
    b[i] = as_type<uint>(v);
}

void WriteUInt(device uint *b, uint i, uint v)
{
    b[i] = v;
}

uint Exchange(device uint *b, uint i, uint v)
{
    return atomic_exchange_explicit((device atomic_uint *)(b + i), v, memory_order_relaxed);
}

float PF(constant uint4 *data, uint i)
{
    return as_type<float>(data[i / 4][i % 4]);
}

int PI(constant uint4 *data, uint i)
{
    return (int)data[i / 4][i % 4];
}

struct Particle
{
    float2 Position;
    float2 Velocity;
    float Density;
};

struct FluidParameters

{

    int Count, GridWidth, GridHeight;

    float2 Minimum, Maximum, Center, Gravity, GridMinimum;

    float Cosine, Sine, Radius, ParticleRadius, InverseDensity, Stiffness, Viscosity, Restitution, TimeStep;
};

FluidParameters FluidParams(constant uint4 *data)

{

    FluidParameters p;

    p.Count = PI(data, 0);
    p.GridWidth = PI(data, 1);
    p.GridHeight = PI(data, 2);

    p.Minimum = float2(PF(data, 4), PF(data, 5));
    p.Maximum = float2(PF(data, 6), PF(data, 7));

    p.Center = float2(PF(data, 8), PF(data, 9));
    p.Gravity = float2(PF(data, 10), PF(data, 11));

    p.GridMinimum = float2(PF(data, 12), PF(data, 13));

    p.Cosine = PF(data, 14);
    p.Sine = PF(data, 15);
    p.Radius = PF(data, 16);
    p.ParticleRadius = PF(data, 17);

    p.InverseDensity = PF(data, 18);
    p.Stiffness = PF(data, 19);
    p.Viscosity = PF(data, 20);

    p.Restitution = PF(data, 21);
    p.TimeStep = PF(data, 22);

    return p;
}

float2 Position(device uint *b, int i)
{
    return float2(ReadFloat(b, 5 * i), ReadFloat(b, 5 * i + 1));
}

Particle LoadParticle(device uint *b, int i)

{

    Particle p;
    p.Position = Position(b, i);

    p.Velocity = float2(ReadFloat(b, 5 * i + 2), ReadFloat(b, 5 * i + 3));
    p.Density = ReadFloat(b, 5 * i + 4);

    return p;
}

void StoreParticle(device uint *b, int i, Particle p)

{

    WriteFloat(b, 5 * i, p.Position.x);
    WriteFloat(b, 5 * i + 1, p.Position.y);

    WriteFloat(b, 5 * i + 2, p.Velocity.x);
    WriteFloat(b, 5 * i + 3, p.Velocity.y);
    WriteFloat(b, 5 * i + 4, p.Density);
}

float2 Rotate(float2 v, float c, float s)
{
    return float2(c * v.x - s * v.y, s * v.x + c * v.y);
}

Particle Constrain(Particle particle, FluidParameters p)

{

    float2 position = Rotate(particle.Position - p.Center, p.Cosine, -p.Sine);

    float2 velocity = Rotate(particle.Velocity, p.Cosine, -p.Sine);

    if (position.x <= p.Minimum.x + p.ParticleRadius)

    {
        position.x = p.Minimum.x + p.ParticleRadius;
        if (velocity.x < 0.0f)
            velocity.x *= -p.Restitution;
    }

    if (position.x >= p.Maximum.x - p.ParticleRadius)

    {
        position.x = p.Maximum.x - p.ParticleRadius;
        if (velocity.x > 0.0f)
            velocity.x *= -p.Restitution;
    }

    if (position.y <= p.Minimum.y + p.ParticleRadius)

    {
        position.y = p.Minimum.y + p.ParticleRadius;
        if (velocity.y < 0.0f)
            velocity.y *= -p.Restitution;
    }

    if (position.y >= p.Maximum.y - p.ParticleRadius)

    {
        position.y = p.Maximum.y - p.ParticleRadius;
        if (velocity.y > 0.0f)
            velocity.y *= -p.Restitution;
    }

    particle.Position = Rotate(position, p.Cosine, p.Sine) + p.Center;

    particle.Velocity = Rotate(velocity, p.Cosine, p.Sine);

    return particle;
}

int2 Cell(float2 position, FluidParameters p)

{

    return clamp(int2(floor((position - p.GridMinimum) / p.Radius)), int2(0, 0),
                 int2(p.GridWidth - 1, p.GridHeight - 1));
}

float3 Kernel(float2 delta, int i, int j, FluidParameters p)

{

    float d2 = dot(delta, delta), inverseRadius2 = 1.0f / (p.Radius * p.Radius);

    if (d2 >= p.Radius * p.Radius || i == j)
        return float3(0.0f, 0.0f, 0.0f);

    if (d2 < 1.0e-12f)

    {

        float sign = i < j ? 1.0f : -1.0f;

        delta = ((i + j) & 1) ? float2(sign * 0.001f * p.Radius, 0.0f) : float2(0.0f, sign * 0.001f * p.Radius);
    }

    float q = 1.0f - d2 * inverseRadius2;

    return float3((-6.0f * inverseRadius2 * q * q) * delta, q * q * q);
}

kernel void ClearHeads(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                       device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                       device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                       constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                       uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                       uint3 gid [[threadgroup_position_in_grid]])

{

    FluidParameters p = FluidParams(constants);

    if (tid.x < (uint)(p.GridWidth * p.GridHeight))
        WriteUInt(b2, tid.x, 0xffffffff);
}

kernel void Predict(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                    device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                    device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                    constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                    uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]])

{

    FluidParameters p = FluidParams(constants);
    int i = (int)tid.x;

    if (i >= p.Count)
        return;

    Particle particle = LoadParticle(b0, i);

    WriteFloat(b5, 2 * i, particle.Position.x);
    WriteFloat(b5, 2 * i + 1, particle.Position.y);

    particle.Velocity += p.TimeStep * p.Gravity;

    particle.Position += p.TimeStep * particle.Velocity;

    StoreParticle(b0, i, Constrain(particle, p));
}

kernel void BuildGrid(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                      device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                      device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                      constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                      uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                      uint3 gid [[threadgroup_position_in_grid]])

{

    FluidParameters p = FluidParams(constants);
    int i = (int)tid.x;

    if (i >= p.Count)
        return;

    int2 cell = Cell(Position(b0, i), p);

    WriteUInt(b3, i, Exchange(b2, cell.y * p.GridWidth + cell.x, i));
}

kernel void DensityLambda(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                          device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                          device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                          constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                          uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                          uint3 gid [[threadgroup_position_in_grid]])

{

    FluidParameters p = FluidParams(constants);
    int i = (int)tid.x;

    if (i >= p.Count)
        return;

    float2 position = Position(b0, i);
    int2 cell = Cell(position, p);

    float density = 1.0f, gradientSquares = 0.0f;
    float2 sum = float2(0.0f, 0.0f);

    for (int y = max(0, cell.y - 1); y <= min(p.GridHeight - 1, cell.y + 1); ++y)

        for (int x = max(0, cell.x - 1); x <= min(p.GridWidth - 1, cell.x + 1); ++x)

            for (int j = (int)ReadUInt(b2, y * p.GridWidth + x); j >= 0; j = (int)ReadUInt(b3, j))

            {

                float3 k = Kernel(position - Position(b0, j), i, j, p);

                density += k.z;
                float2 gradient = p.InverseDensity * k.xy;

                sum += gradient;
                gradientSquares += dot(gradient, gradient);
            }

    WriteFloat(b0, 5 * i + 4, density);

    WriteFloat(b4, i,
               -max(density * p.InverseDensity - 1.0f, 0.0f) /

                   (gradientSquares + dot(sum, sum) + 0.01f / (p.Radius * p.Radius)));
}

kernel void Correct(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                    device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                    device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                    constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                    uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]])

{

    FluidParameters p = FluidParams(constants);
    int i = (int)tid.x;

    if (i >= p.Count)
        return;

    Particle particle = LoadParticle(b0, i);
    int2 cell = Cell(particle.Position, p);

    float2 correction = float2(0.0f, 0.0f);

    for (int y = max(0, cell.y - 1); y <= min(p.GridHeight - 1, cell.y + 1); ++y)

        for (int x = max(0, cell.x - 1); x <= min(p.GridWidth - 1, cell.x + 1); ++x)

            for (int j = (int)ReadUInt(b2, y * p.GridWidth + x); j >= 0; j = (int)ReadUInt(b3, j))

                correction += (p.Stiffness * (ReadFloat(b4, i) + ReadFloat(b4, j)) * p.InverseDensity) *

                              Kernel(particle.Position - Position(b0, j), i, j, p).xy;

    particle.Position += min(1.0f, 0.2f * p.Radius / max(length(correction), 1.0e-6f)) * correction;

    StoreParticle(b1, i, Constrain(particle, p));
}

kernel void Reconstruct(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                        device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                        device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                        constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                        uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                        uint3 gid [[threadgroup_position_in_grid]])

{

    FluidParameters p = FluidParams(constants);
    int i = (int)tid.x;

    if (i >= p.Count)
        return;

    float2 velocity = (Position(b0, i) - float2(ReadFloat(b5, 2 * i), ReadFloat(b5, 2 * i + 1))) / p.TimeStep;

    WriteFloat(b0, 5 * i + 2, velocity.x);
    WriteFloat(b0, 5 * i + 3, velocity.y);
}

kernel void Viscosity(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                      device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                      device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                      constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                      uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                      uint3 gid [[threadgroup_position_in_grid]])

{

    FluidParameters p = FluidParams(constants);
    int i = (int)tid.x;

    if (i >= p.Count)
        return;

    Particle particle = LoadParticle(b0, i);
    int2 cell = Cell(particle.Position, p);

    float2 exchange = float2(0.0f, 0.0f);

    for (int y = max(0, cell.y - 1); y <= min(p.GridHeight - 1, cell.y + 1); ++y)

        for (int x = max(0, cell.x - 1); x <= min(p.GridWidth - 1, cell.x + 1); ++x)

            for (int j = (int)ReadUInt(b2, y * p.GridWidth + x); j >= 0; j = (int)ReadUInt(b3, j))

            {

                Particle other = LoadParticle(b0, j);

                float weight = Kernel(particle.Position - other.Position, i, j, p).z;

                exchange += (p.Viscosity * weight / max(particle.Density, other.Density)) *
                            (other.Velocity - particle.Velocity);
            }

    particle.Velocity += exchange;
    StoreParticle(b1, i, Constrain(particle, p));
}
)SHADER"
