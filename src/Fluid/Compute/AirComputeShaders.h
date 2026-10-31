#pragma once

namespace Riemann2D::Compute
{
    inline constexpr const char* AirShaders = R"SHADER(
struct AirParameters
{
    int Width, Height, Component, MacCormack;
    float InverseX, InverseY, TimeStep, Diffusion, Vorticity, Decay;
};
AirParameters AirParams(DATA)
{
    AirParameters p;
    p.Width = PI(data, 0); p.Height = PI(data, 1); p.Component = PI(data, 2);
    p.InverseX = PF(data, 4); p.InverseY = PF(data, 5); p.TimeStep = PF(data, 6);
    p.Diffusion = PF(data, 7); p.Vorticity = PF(data, 8); p.Decay = PF(data, 9); p.MacCormack = PI(data, 10);
    return p;
}
struct Sample { int A, B, C, D; float X, Y; };
Sample GridSample(int w, int h, float x, float y)
{
    x = clamp(x, 0.0f, (float)(w - 1)); y = clamp(y, 0.0f, (float)(h - 1));
    int ix = (int)x, iy = (int)y, nx = min(ix + 1, w - 1), ny = min(iy + 1, h - 1);
    Sample s; s.A = iy * w + ix; s.B = iy * w + nx; s.C = ny * w + ix; s.D = ny * w + nx;
    s.X = x - ix; s.Y = y - iy; return s;
}
float SampleRead(BUFFER f, Sample s)
{
    float a = ReadFloat(f, s.A), b = ReadFloat(f, s.B), c = ReadFloat(f, s.C), d = ReadFloat(f, s.D);
    return (a + s.X * (b - a)) * (1.0f - s.Y) + (c + s.X * (d - c)) * s.Y;
}
float SampleLimit(BUFFER f, Sample s, float value)
{
    float a = ReadFloat(f, s.A), b = ReadFloat(f, s.B), c = ReadFloat(f, s.C), d = ReadFloat(f, s.D);
    return clamp(value, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
}
float2 Velocity(BUFFER u, BUFFER v, AirParameters p, float x, float y)
{
    return float2(SampleRead(u, GridSample(p.Width + 1, p.Height, x, y - 0.5f)),
        SampleRead(v, GridSample(p.Width, p.Height + 1, x - 0.5f, y)));
}
float2 Center(BUFFER u, BUFFER v, int x, int y, int w)
{
    return float2(0.5f * (ReadFloat(u, y * (w + 1) + x) + ReadFloat(u, y * (w + 1) + x + 1)),
        0.5f * (ReadFloat(v, y * w + x) + ReadFloat(v, (y + 1) * w + x)));
}
float AdvectedFace(BUFFER source, BUFFER u, BUFFER v, AirParameters p, int i)
{
    int w = p.Width + (p.Component == 0 ? 1 : 0), h = p.Height + (p.Component == 1 ? 1 : 0);
    int x = i % w, y = i / w;
    if ((p.Component == 0 && (x == 0 || x == w - 1)) || (p.Component == 1 && (y == 0 || y == h - 1))) return 0.0f;
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
KERNEL(AdvectFaces)
{
    AirParameters p = AirParams(constants); int i = (int)tid.x;
    int count = p.Component == 0 ? (p.Width + 1) * p.Height : p.Width * (p.Height + 1);
    if (i >= count) return;
    float value = p.Component == 0 ? AdvectedFace(b0, b0, b1, p, i) : AdvectedFace(b1, b0, b1, p, i);
    WriteFloat(b2, i, value);
}
KERNEL(ComputeCurl)
{
    AirParameters p = AirParams(constants); int i = (int)tid.x, w = p.Width, h = p.Height;
    if (i >= w * h) return;
    int x = i % w, y = i / w; float value = 0.0f;
    if (x > 0 && x + 1 < w && y > 0 && y + 1 < h)
        value = 0.5f * ((Center(b0, b1, x + 1, y, w).y - Center(b0, b1, x - 1, y, w).y) * p.InverseX -
            (Center(b0, b1, x, y + 1, w).x - Center(b0, b1, x, y - 1, w).x) * p.InverseY);
    WriteFloat(b2, i, value);
}
KERNEL(ComputeForce)
{
    AirParameters p = AirParams(constants); int i = (int)tid.x, w = p.Width, h = p.Height;
    if (i >= w * h) return;
    int x = i % w, y = i / w; float2 value = float2(0.0f, 0.0f);
    if (p.Vorticity > 0.0f && x > 0 && x + 1 < w && y > 0 && y + 1 < h)
    {
        float2 gradient = float2((abs(ReadFloat(b0, i + 1)) - abs(ReadFloat(b0, i - 1))) * p.InverseX,
            (abs(ReadFloat(b0, i + w)) - abs(ReadFloat(b0, i - w))) * p.InverseY);
        float scale = p.Vorticity * min(1.0f / p.InverseX, 1.0f / p.InverseY) * ReadFloat(b0, i) / max(length(gradient), 1.0e-5f);
        value = scale * float2(gradient.y, -gradient.x);
    }
    WriteFloat(b1, 2 * i, value.x); WriteFloat(b1, 2 * i + 1, value.y);
}
KERNEL(ApplyForce)
{
    AirParameters p = AirParams(constants); int i = (int)tid.x;
    if (i < (p.Width + 1) * p.Height)
    {
        int x = i % (p.Width + 1), y = i / (p.Width + 1);
        if (x > 0 && x < p.Width) WriteFloat(b0, i, ReadFloat(b0, i) + 0.5f * p.TimeStep *
            (ReadFloat(b2, 2 * (y * p.Width + x - 1)) + ReadFloat(b2, 2 * (y * p.Width + x))));
    }
    if (i < p.Width * (p.Height + 1))
    {
        int x = i % p.Width, y = i / p.Width;
        if (y > 0 && y < p.Height) WriteFloat(b1, i, ReadFloat(b1, i) + 0.5f * p.TimeStep *
            (ReadFloat(b2, 2 * ((y - 1) * p.Width + x) + 1) + ReadFloat(b2, 2 * (y * p.Width + x) + 1)));
    }
}
KERNEL(PreparePressure)
{
    AirParameters p = AirParams(constants); int i = (int)tid.x;
    if (i >= p.Width * p.Height) return;
    int x = i % p.Width, y = i / p.Width;
    WriteFloat(b2, i, 0.0f);
    WriteFloat(b3, i, -(ReadFloat(b0, y * (p.Width + 1) + x + 1) - ReadFloat(b0, y * (p.Width + 1) + x)) * p.InverseX -
        (ReadFloat(b1, (y + 1) * p.Width + x) - ReadFloat(b1, y * p.Width + x)) * p.InverseY);
}
struct Level { int Width, Height; float WeightX, WeightY; };
Level LevelParams(DATA)
{
    Level l; l.Width = PI(data, 16); l.Height = PI(data, 17); l.WeightX = PF(data, 18); l.WeightY = PF(data, 19); return l;
}
float NeighborSum(BUFFER pressure, Level l, int x, int y, int i)
{
    return ((x > 0 ? ReadFloat(pressure, i - 1) : 0.0f) + (x + 1 < l.Width ? ReadFloat(pressure, i + 1) : 0.0f)) * l.WeightX +
        ((y > 0 ? ReadFloat(pressure, i - l.Width) : 0.0f) + (y + 1 < l.Height ? ReadFloat(pressure, i + l.Width) : 0.0f)) * l.WeightY;
}
KERNEL(Smooth)
{
    Level l = LevelParams(constants); int i = (int)tid.x;
    if (i >= l.Width * l.Height) return;
    int x = i % l.Width, y = i / l.Width;
    if (((x + y) & 1) != PI(constants, 3)) return;
    float old = ReadFloat(b0, i);
    WriteFloat(b0, i, old + 1.15f * ((NeighborSum(b0, l, x, y, i) + ReadFloat(b1, i)) * ReadFloat(b2, i) - old));
}
KERNEL(SmoothCoarsest)
{
    Level l = LevelParams(constants); int i = (int)lid.x, count = l.Width * l.Height;
    if (i < count) shared0[i] = ReadFloat(b0, i);
    SHARED_BARRIER;
    for (int iteration = 0; iteration < 40; ++iteration)
        for (int parity = 0; parity < 2; ++parity)
        {
            if (i < count)
            {
                int x = i % l.Width, y = i / l.Width;
                if (((x + y) & 1) == parity)
                {
                    float sum = ((x > 0 ? shared0[i - 1] : 0.0f) + (x + 1 < l.Width ? shared0[i + 1] : 0.0f)) * l.WeightX +
                        ((y > 0 ? shared0[i - l.Width] : 0.0f) + (y + 1 < l.Height ? shared0[i + l.Width] : 0.0f)) * l.WeightY;
                    shared0[i] += 1.15f * ((sum + ReadFloat(b1, i)) * ReadFloat(b2, i) - shared0[i]);
                }
            }
            SHARED_BARRIER;
        }
    if (i < count) WriteFloat(b0, i, shared0[i]);
}
KERNEL(Residual)
{
    Level l = LevelParams(constants); int i = (int)tid.x;
    if (i < l.Width * l.Height) WriteFloat(b3, i, ReadFloat(b1, i) - ReadFloat(b0, i) / ReadFloat(b2, i) +
        NeighborSum(b0, l, i % l.Width, i / l.Width, i));
}
KERNEL(Restrict)
{
    Level fine = LevelParams(constants); int w = PI(constants, 20), h = PI(constants, 21), i = (int)tid.x;
    float value = 0.0f;
    if (i < w * h)
    {
        int x = i % w, y = i / w, count = 0;
        for (int dy = 0; dy < 2 && 2 * y + dy < fine.Height; ++dy)
            for (int dx = 0; dx < 2 && 2 * x + dx < fine.Width; ++dx)
            { value += ReadFloat(b0, (2 * y + dy) * fine.Width + 2 * x + dx); ++count; }
        value /= count; WriteFloat(b1, i, 0.0f); WriteFloat(b2, i, value);
    }
    shared0[lid.x] = value; SHARED_BARRIER;
    for (uint stride = 128; stride > 0; stride /= 2)
    {
        if (lid.x < stride) shared0[lid.x] += shared0[lid.x + stride];
        SHARED_BARRIER;
    }
    if (lid.x == 0) WriteFloat(b3, gid.x, shared0[0]);
}
// Two-stage deterministic reduction avoids unsupported floating-point atomics on DX11.
KERNEL(ReduceMean)
{
    int count = PI(constants, 20) * PI(constants, 21), blocks = (count + 255) / 256;
    float sum = 0.0f;
    for (int i = (int)lid.x; i < blocks; i += 256) sum += ReadFloat(b0, i);
    shared0[lid.x] = sum; SHARED_BARRIER;
    for (uint stride = 128; stride > 0; stride /= 2)
    {
        if (lid.x < stride) shared0[lid.x] += shared0[lid.x + stride];
        SHARED_BARRIER;
    }
    if (lid.x == 0) WriteFloat(b1, 0, shared0[0] / count);
}
KERNEL(RemoveMean)
{
    int count = PI(constants, 20) * PI(constants, 21);
    if ((int)tid.x < count) WriteFloat(b0, tid.x, ReadFloat(b0, tid.x) - ReadFloat(b1, 0));
}
KERNEL(Prolong)
{
    Level fine = LevelParams(constants); int w = PI(constants, 20), h = PI(constants, 21), i = (int)tid.x;
    if (i >= fine.Width * fine.Height) return;
    float value = SampleRead(b1, GridSample(w, h, (i % fine.Width + 0.5f) * w / fine.Width - 0.5f,
        (i / fine.Width + 0.5f) * h / fine.Height - 0.5f));
    WriteFloat(b0, i, ReadFloat(b0, i) + value);
}
KERNEL(ApplyPressure)
{
    AirParameters p = AirParams(constants); int i = (int)tid.x;
    if (i < (p.Width + 1) * p.Height)
    {
        int x = i % (p.Width + 1), y = i / (p.Width + 1); float value = 0.0f;
        if (x > 0 && x < p.Width) value = ReadFloat(b0, i) -
            (ReadFloat(b2, y * p.Width + x) - ReadFloat(b2, y * p.Width + x - 1)) * p.InverseX;
        WriteFloat(b0, i, value);
    }
    if (i < p.Width * (p.Height + 1))
    {
        int x = i % p.Width, y = i / p.Width; float value = 0.0f;
        if (y > 0 && y < p.Height) value = ReadFloat(b1, i) -
            (ReadFloat(b2, y * p.Width + x) - ReadFloat(b2, (y - 1) * p.Width + x)) * p.InverseY;
        WriteFloat(b1, i, value);
    }
}
KERNEL(ForwardDensity)
{
    AirParameters p = AirParams(constants); int i = (int)tid.x;
    if (i >= p.Width * p.Height) return;
    int x = i % p.Width, y = i / p.Width; float2 velocity = Center(b1, b2, x, y, p.Width);
    WriteFloat(b4, 2 * i, velocity.x); WriteFloat(b4, 2 * i + 1, velocity.y);
    WriteFloat(b3, i, SampleRead(b0, GridSample(p.Width, p.Height, x - p.TimeStep * p.InverseX * velocity.x,
        y - p.TimeStep * p.InverseY * velocity.y)));
}
KERNEL(CorrectDensity)
{
    AirParameters p = AirParams(constants); int i = (int)tid.x;
    if (i >= p.Width * p.Height) return;
    int x = i % p.Width, y = i / p.Width;
    float dx = p.TimeStep * p.InverseX * ReadFloat(b2, 2 * i), dy = p.TimeStep * p.InverseY * ReadFloat(b2, 2 * i + 1);
    float value = ReadFloat(b1, i);
    if (p.MacCormack != 0)
    {
        float reverse = SampleRead(b1, GridSample(p.Width, p.Height, x + dx, y + dy));
        value = SampleLimit(b0, GridSample(p.Width, p.Height, x - dx, y - dy), value + 0.5f * (ReadFloat(b0, i) - reverse));
    }
    WriteFloat(b3, i, p.Decay * max(0.0f, value));
}
)SHADER";
}
