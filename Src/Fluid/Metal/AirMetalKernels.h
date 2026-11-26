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

struct AirParameters

{

    int Width, Height, Component, MacCormack;

    float InverseX, InverseY, TimeStep, Diffusion, Vorticity, Decay;
};

AirParameters AirParams(constant uint4 *data)

{

    AirParameters p;

    p.Width = PI(data, 0);
    p.Height = PI(data, 1);
    p.Component = PI(data, 2);

    p.InverseX = PF(data, 4);
    p.InverseY = PF(data, 5);
    p.TimeStep = PF(data, 6);

    p.Diffusion = PF(data, 7);
    p.Vorticity = PF(data, 8);
    p.Decay = PF(data, 9);
    p.MacCormack = PI(data, 10);

    return p;
}

struct Sample
{
    int A, B, C, D;
    float X, Y;
};

Sample GridSample(int w, int h, float x, float y)

{

    x = clamp(x, 0.0f, (float)(w - 1));
    y = clamp(y, 0.0f, (float)(h - 1));

    int ix = (int)x, iy = (int)y, nx = min(ix + 1, w - 1), ny = min(iy + 1, h - 1);

    Sample s;
    s.A = iy * w + ix;
    s.B = iy * w + nx;
    s.C = ny * w + ix;
    s.D = ny * w + nx;

    s.X = x - ix;
    s.Y = y - iy;
    return s;
}

float SampleRead(device uint *f, Sample s)

{

    float a = ReadFloat(f, s.A), b = ReadFloat(f, s.B), c = ReadFloat(f, s.C), d = ReadFloat(f, s.D);

    return (a + s.X * (b - a)) * (1.0f - s.Y) + (c + s.X * (d - c)) * s.Y;
}

float SampleLimit(device uint *f, Sample s, float value)

{

    float a = ReadFloat(f, s.A), b = ReadFloat(f, s.B), c = ReadFloat(f, s.C), d = ReadFloat(f, s.D);

    return clamp(value, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
}

float2 Velocity(device uint *u, device uint *v, AirParameters p, float x, float y)

{

    return float2(SampleRead(u, GridSample(p.Width + 1, p.Height, x, y - 0.5f)),

                  SampleRead(v, GridSample(p.Width, p.Height + 1, x - 0.5f, y)));
}

float2 Center(device uint *u, device uint *v, int x, int y, int w)

{

    return float2(0.5f * (ReadFloat(u, y * (w + 1) + x) + ReadFloat(u, y * (w + 1) + x + 1)),

                  0.5f * (ReadFloat(v, y * w + x) + ReadFloat(v, (y + 1) * w + x)));
}

float AdvectedFace(device uint *source, device uint *u, device uint *v, AirParameters p, int i)

{

    int w = p.Width + (p.Component == 0 ? 1 : 0), h = p.Height + (p.Component == 1 ? 1 : 0);

    int x = i % w, y = i / w;

    if ((p.Component == 0 && (x == 0 || x == w - 1)) || (p.Component == 1 && (y == 0 || y == h - 1)))
        return 0.0f;

    float gx = x + (p.Component == 1 ? 0.5f : 0.0f), gy = y + (p.Component == 0 ? 0.5f : 0.0f);

    float2 velocity = Velocity(u, v, p, gx, gy);

    float dtX = p.TimeStep * p.InverseX, dtY = p.TimeStep * p.InverseY;

    float2 midpoint = Velocity(u, v, p, gx - 0.5f * dtX * velocity.x, gy - 0.5f * dtY * velocity.y);

    float value = SampleRead(source, GridSample(w, h, x - dtX * midpoint.x, y - dtY * midpoint.y));

    float ax = p.Diffusion * p.TimeStep * p.InverseX * p.InverseX;

    float ay = p.Diffusion * p.TimeStep * p.InverseY * p.InverseY;

    float scale = min(1.0f, 0.49f / max(ax + ay, 1.0e-6f));

    if (x > 0 && x + 1 < w && y > 0 && y + 1 < h)

        value += scale * (ax * (ReadFloat(source, i - 1) + ReadFloat(source, i + 1) - 2.0f * ReadFloat(source, i)) +

                          ay * (ReadFloat(source, i - w) + ReadFloat(source, i + w) - 2.0f * ReadFloat(source, i)));

    return value;
}

kernel void AdvectFaces(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                        device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                        device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                        constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                        uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                        uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x;

    int count = p.Component == 0 ? (p.Width + 1) * p.Height : p.Width * (p.Height + 1);

    if (i >= count)
        return;

    float value = p.Component == 0 ? AdvectedFace(b0, b0, b1, p, i) : AdvectedFace(b1, b0, b1, p, i);

    WriteFloat(b2, i, value);
}

kernel void ComputeCurl(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                        device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                        device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                        constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                        uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                        uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x, w = p.Width, h = p.Height;

    if (i >= w * h)
        return;

    int x = i % w, y = i / w;
    float value = 0.0f;

    if (x > 0 && x + 1 < w && y > 0 && y + 1 < h)

        value = 0.5f * ((Center(b0, b1, x + 1, y, w).y - Center(b0, b1, x - 1, y, w).y) * p.InverseX -

                        (Center(b0, b1, x, y + 1, w).x - Center(b0, b1, x, y - 1, w).x) * p.InverseY);

    WriteFloat(b2, i, value);
}

kernel void ComputeForce(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                         device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                         device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                         constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                         uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                         uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x, w = p.Width, h = p.Height;

    if (i >= w * h)
        return;

    int x = i % w, y = i / w;
    float2 value = float2(0.0f, 0.0f);

    if (p.Vorticity > 0.0f && x > 0 && x + 1 < w && y > 0 && y + 1 < h)

    {

        float2 gradient = float2((abs(ReadFloat(b0, i + 1)) - abs(ReadFloat(b0, i - 1))) * p.InverseX,

                                 (abs(ReadFloat(b0, i + w)) - abs(ReadFloat(b0, i - w))) * p.InverseY);

        float scale =
            p.Vorticity * min(1.0f / p.InverseX, 1.0f / p.InverseY) * ReadFloat(b0, i) / max(length(gradient), 1.0e-5f);

        value = scale * float2(gradient.y, -gradient.x);
    }

    WriteFloat(b1, 2 * i, value.x);
    WriteFloat(b1, 2 * i + 1, value.y);
}

kernel void ApplyForce(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                       device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                       device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                       constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                       uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                       uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x;

    if (i < (p.Width + 1) * p.Height)

    {

        int x = i % (p.Width + 1), y = i / (p.Width + 1);

        if (x > 0 && x < p.Width)
            WriteFloat(b0, i,
                       ReadFloat(b0, i) +
                           0.5f * p.TimeStep *

                               (ReadFloat(b2, 2 * (y * p.Width + x - 1)) + ReadFloat(b2, 2 * (y * p.Width + x))));
    }

    if (i < p.Width * (p.Height + 1))

    {

        int x = i % p.Width, y = i / p.Width;

        if (y > 0 && y < p.Height)
            WriteFloat(b1, i,
                       ReadFloat(b1, i) + 0.5f * p.TimeStep *

                                              (ReadFloat(b2, 2 * ((y - 1) * p.Width + x) + 1) +
                                               ReadFloat(b2, 2 * (y * p.Width + x) + 1)));
    }
}

kernel void PreparePressure(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                            device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                            device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                            constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                            uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                            uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x;

    if (i >= p.Width * p.Height)
        return;

    if ((int)ReadUInt(b4, i) >= 0)
    {
        WriteFloat(b2, i, 0);
        WriteFloat(b3, i, 0);
        return;
    }

    int x = i % p.Width, y = i / p.Width;

    WriteFloat(b2, i, 0.0f);

    WriteFloat(b3, i,
               -(ReadFloat(b0, y * (p.Width + 1) + x + 1) - ReadFloat(b0, y * (p.Width + 1) + x)) * p.InverseX -

                   (ReadFloat(b1, (y + 1) * p.Width + x) - ReadFloat(b1, y * p.Width + x)) * p.InverseY);
}

struct Level
{
    int Width, Height;
    float WeightX, WeightY;
};

Level LevelParams(constant uint4 *data)

{

    Level l;
    l.Width = PI(data, 16);
    l.Height = PI(data, 17);
    l.WeightX = PF(data, 18);
    l.WeightY = PF(data, 19);
    return l;
}

float NeighborSum(device uint *pressure, device uint *solid, Level l, int x, int y, int i)

{

    return ((x > 0 && (int)ReadUInt(solid, i - 1) < 0 ? ReadFloat(pressure, i - 1) : 0.0f) +
            (x + 1 < l.Width && (int)ReadUInt(solid, i + 1) < 0 ? ReadFloat(pressure, i + 1) : 0.0f)) *
               l.WeightX +

           ((y > 0 && (int)ReadUInt(solid, i - l.Width) < 0 ? ReadFloat(pressure, i - l.Width) : 0.0f) +
            (y + 1 < l.Height && (int)ReadUInt(solid, i + l.Width) < 0 ? ReadFloat(pressure, i + l.Width) : 0.0f)) *
               l.WeightY;
}

kernel void Smooth(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                   device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                   device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                   constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                   uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                   uint3 gid [[threadgroup_position_in_grid]])

{

    Level l = LevelParams(constants);
    int i = (int)tid.x;

    if (i >= l.Width * l.Height)
        return;

    int x = i % l.Width, y = i / l.Width;

    if (((x + y) & 1) != PI(constants, 3))
        return;

    float old = ReadFloat(b0, i);

    WriteFloat(b0, i, old + 1.15f * ((NeighborSum(b0, b4, l, x, y, i) + ReadFloat(b1, i)) * ReadFloat(b2, i) - old));
}

kernel void SmoothCoarsest(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                           device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                           device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                           constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                           uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                           uint3 gid [[threadgroup_position_in_grid]])

{

    Level l = LevelParams(constants);
    int i = (int)lid.x, count = l.Width * l.Height;

    if (i < count)
        shared0[i] = ReadFloat(b0, i);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int iteration = 0; iteration < 40; ++iteration)

        for (int parity = 0; parity < 2; ++parity)

        {

            if (i < count)

            {

                int x = i % l.Width, y = i / l.Width;

                if (((x + y) & 1) == parity)

                {

                    float sum =
                        ((x > 0 && (int)ReadUInt(b4, i - 1) < 0 ? shared0[i - 1] : 0.0f) +
                         (x + 1 < l.Width && (int)ReadUInt(b4, i + 1) < 0 ? shared0[i + 1] : 0.0f)) *
                            l.WeightX +

                        ((y > 0 && (int)ReadUInt(b4, i - l.Width) < 0 ? shared0[i - l.Width] : 0.0f) +
                         (y + 1 < l.Height && (int)ReadUInt(b4, i + l.Width) < 0 ? shared0[i + l.Width] : 0.0f)) *
                            l.WeightY;

                    shared0[i] += 1.15f * ((sum + ReadFloat(b1, i)) * ReadFloat(b2, i) - shared0[i]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

    if (i < count)
        WriteFloat(b0, i, shared0[i]);
}

kernel void Residual(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                     device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                     device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                     constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                     uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                     uint3 gid [[threadgroup_position_in_grid]])

{

    Level l = LevelParams(constants);
    int i = (int)tid.x;

    if (i < l.Width * l.Height && ReadFloat(b2, i) == 0)
    {
        WriteFloat(b3, i, 0);
        return;
    }

    if (i < l.Width * l.Height)
        WriteFloat(b3, i,
                   ReadFloat(b1, i) - ReadFloat(b0, i) / ReadFloat(b2, i) +

                       NeighborSum(b0, b4, l, i % l.Width, i / l.Width, i));
}

kernel void Restrict(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                     device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                     device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                     constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                     uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                     uint3 gid [[threadgroup_position_in_grid]])

{

    Level fine = LevelParams(constants);
    int w = PI(constants, 20), h = PI(constants, 21), i = (int)tid.x;

    float value = 0.0f;

    if (i < w * h)

    {

        int x = i % w, y = i / w, count = 0;

        for (int dy = 0; dy < 2 && 2 * y + dy < fine.Height; ++dy)

            for (int dx = 0; dx < 2 && 2 * x + dx < fine.Width; ++dx)

            {
                value += ReadFloat(b0, (2 * y + dy) * fine.Width + 2 * x + dx);
                ++count;
            }

        value /= count;
        WriteFloat(b1, i, 0.0f);
        WriteFloat(b2, i, value);
    }

    shared0[lid.x] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride /= 2)

    {

        if (lid.x < stride)
            shared0[lid.x] += shared0[lid.x + stride];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid.x == 0)
        WriteFloat(b3, gid.x, shared0[0]);
}

kernel void ReduceMean(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                       device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                       device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                       constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                       uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                       uint3 gid [[threadgroup_position_in_grid]])

{

    int count = PI(constants, 20) * PI(constants, 21), blocks = (count + 255) / 256;

    float sum = 0.0f;

    for (int i = (int)lid.x; i < blocks; i += 256)
        sum += ReadFloat(b0, i);

    shared0[lid.x] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride /= 2)

    {

        if (lid.x < stride)
            shared0[lid.x] += shared0[lid.x + stride];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid.x == 0)
        WriteFloat(b1, 0, shared0[0] / count);
}

kernel void RemoveMean(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                       device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                       device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                       constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                       uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                       uint3 gid [[threadgroup_position_in_grid]])

{

    int count = PI(constants, 20) * PI(constants, 21);

    if ((int)tid.x < count)
        WriteFloat(b0, tid.x, ReadFloat(b0, tid.x) - ReadFloat(b1, 0));
}

kernel void Prolong(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                    device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                    device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                    constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                    uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                    uint3 gid [[threadgroup_position_in_grid]])

{

    Level fine = LevelParams(constants);
    int w = PI(constants, 20), h = PI(constants, 21), i = (int)tid.x;

    if (i >= fine.Width * fine.Height || (int)ReadUInt(b2, i) >= 0)
        return;

    float value = SampleRead(b1, GridSample(w, h, (i % fine.Width + 0.5f) * w / fine.Width - 0.5f,

                                            (i / fine.Width + 0.5f) * h / fine.Height - 0.5f));

    WriteFloat(b0, i, ReadFloat(b0, i) + value);
}

kernel void ApplyPressure(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                          device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                          device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                          constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                          uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                          uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x;

    if (i < (p.Width + 1) * p.Height)

    {

        int x = i % (p.Width + 1), y = i / (p.Width + 1);
        float value = 0.0f;

        if (x > 0 && x < p.Width)
            value = ReadFloat(b0, i) -

                    (ReadFloat(b2, y * p.Width + x) - ReadFloat(b2, y * p.Width + x - 1)) * p.InverseX;

        WriteFloat(b0, i, value);
    }

    if (i < p.Width * (p.Height + 1))

    {

        int x = i % p.Width, y = i / p.Width;
        float value = 0.0f;

        if (y > 0 && y < p.Height)
            value = ReadFloat(b1, i) -

                    (ReadFloat(b2, y * p.Width + x) - ReadFloat(b2, (y - 1) * p.Width + x)) * p.InverseY;

        WriteFloat(b1, i, value);
    }
}

float2 ClipTrace(float2 start, float2 end, device uint *solid, int w, int h)

{

    float2 delta = end - start, previous = start;

    int steps = max(1, (int)ceil(max(abs(delta.x), abs(delta.y)) * 3));

    for (int i = 1; i <= steps; ++i)

    {

        float2 p = start + ((float)i / steps) * delta;

        int x = (int)floor(p.x + .5f), y = (int)floor(p.y + .5f);

        if (x >= 0 && x < w && y >= 0 && y < h && (int)ReadUInt(solid, y * w + x) >= 0)
            return previous;

        previous = p;
    }

    return end;
}

kernel void CoarsenMask(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                        device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                        device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                        constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                        uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                        uint3 gid [[threadgroup_position_in_grid]])

{

    Level l = LevelParams(constants);
    int i = (int)tid.x;

    if (i >= l.Width * l.Height)
        return;

    int w = PI(constants, 20), h = PI(constants, 21), x = 2 * (i % l.Width), y = 2 * (i / l.Width), solid = 0;

    for (int dy = 0; dy < 2 && y + dy < h; ++dy)

        for (int dx = 0; dx < 2 && x + dx < w; ++dx)

            if ((int)ReadUInt(b0, (y + dy) * w + x + dx) < 0)
                solid = -1;

    WriteUInt(b1, i, (uint)solid);
}

kernel void MaskDiagonal(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                         device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                         device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                         constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                         uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                         uint3 gid [[threadgroup_position_in_grid]])

{

    Level l = LevelParams(constants);
    int i = (int)tid.x;

    if (i >= l.Width * l.Height)
        return;

    int x = i % l.Width, y = i / l.Width;

    float diagonal = 0;

    if ((int)ReadUInt(b0, i) < 0)

    {

        if (x > 0 && (int)ReadUInt(b0, i - 1) < 0)
            diagonal += l.WeightX;

        if (x + 1 < l.Width && (int)ReadUInt(b0, i + 1) < 0)
            diagonal += l.WeightX;

        if (y > 0 && (int)ReadUInt(b0, i - l.Width) < 0)
            diagonal += l.WeightY;

        if (y + 1 < l.Height && (int)ReadUInt(b0, i + l.Width) < 0)
            diagonal += l.WeightY;
    }

    WriteFloat(b1, i, diagonal > 0 ? 1.0f / diagonal : 0);
}

kernel void DisplaceDensity(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                            device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                            device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                            constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                            uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                            uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x;

    if (i >= p.Width * p.Height)
        return;

    float value = (int)ReadUInt(b2, i) < 0 ? ReadFloat(b0, i) : 0;

    for (int source = (int)ReadUInt(b3, i); source >= 0; source = (int)ReadUInt(b4, source))
        value += ReadFloat(b0, source);

    WriteFloat(b1, i, value);
}

kernel void EnforceSolid(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                         device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                         device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                         constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                         uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                         uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x;

    if (i < (p.Width + 1) * p.Height && (int)ReadUInt(b2, i) >= 0)
        WriteFloat(b0, i, ReadFloat(b4, i));

    if (i < p.Width * (p.Height + 1) && (int)ReadUInt(b3, i) >= 0)
        WriteFloat(b1, i, ReadFloat(b5, i));
}

kernel void ForwardDensity(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                           device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                           device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                           constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                           uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                           uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x;

    if (i >= p.Width * p.Height)
        return;

    int x = i % p.Width, y = i / p.Width;
    float2 velocity = Center(b1, b2, x, y, p.Width);

    WriteFloat(b4, 2 * i, velocity.x);
    WriteFloat(b4, 2 * i + 1, velocity.y);

    float2 donor = ClipTrace(float2(x, y),
                             float2(x - p.TimeStep * p.InverseX * velocity.x, y - p.TimeStep * p.InverseY * velocity.y),
                             b5, p.Width, p.Height);

    WriteFloat(b3, i, (int)ReadUInt(b5, i) >= 0 ? 0 : SampleRead(b0, GridSample(p.Width, p.Height, donor.x, donor.y)));
}

kernel void CorrectDensity(device uint *b0 [[buffer(0)]], device uint *b1 [[buffer(1)]], device uint *b2 [[buffer(2)]],
                           device uint *b3 [[buffer(3)]], device uint *b4 [[buffer(4)]], device uint *b5 [[buffer(5)]],
                           device uint *b6 [[buffer(6)]], device uint *b7 [[buffer(7)]],
                           constant uint4 *constants [[buffer(8)]], threadgroup float *shared0 [[threadgroup(0)]],
                           uint3 tid [[thread_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                           uint3 gid [[threadgroup_position_in_grid]])

{

    AirParameters p = AirParams(constants);
    int i = (int)tid.x;

    if (i >= p.Width * p.Height)
        return;

    int x = i % p.Width, y = i / p.Width;

    float dx = p.TimeStep * p.InverseX * ReadFloat(b2, 2 * i), dy = p.TimeStep * p.InverseY * ReadFloat(b2, 2 * i + 1);

    float value = ReadFloat(b1, i);

    if (p.MacCormack != 0)

    {

        float2 ahead = ClipTrace(float2(x, y), float2(x + dx, y + dy), b4, p.Width, p.Height);

        float2 back = ClipTrace(float2(x, y), float2(x - dx, y - dy), b4, p.Width, p.Height);

        float reverse = SampleRead(b1, GridSample(p.Width, p.Height, ahead.x, ahead.y));

        value =
            SampleLimit(b0, GridSample(p.Width, p.Height, back.x, back.y), value + 0.5f * (ReadFloat(b0, i) - reverse));
    }

    WriteFloat(b3, i, (int)ReadUInt(b4, i) >= 0 ? 0 : p.Decay * max(0.0f, value));
}
)SHADER"
