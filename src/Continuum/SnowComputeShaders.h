#pragma once

namespace Riemann2D::Compute
{
    // This source is compiled as native HLSL or Metal by the shared compute device.
    inline constexpr const char* SnowShaders = R"MPM(
float4 MIdentity() { return float4(1,0,0,1); }
float4 MTranspose(float4 a) { return a.xzyw; }
float4 MMultiply(float4 a, float4 b) {
    return float4(a.x*b.x+a.y*b.z, a.x*b.y+a.y*b.w, a.z*b.x+a.w*b.z, a.z*b.y+a.w*b.w);
}
float2 MApply(float4 a, float2 v) { return float2(a.x*v.x+a.y*v.y, a.z*v.x+a.w*v.y); }
float MDet(float4 a) { return a.x*a.w-a.y*a.z; }
float4 MOuter(float2 a, float2 b) { return float4(a.x*b.x,a.x*b.y,a.y*b.x,a.y*b.y); }
float2 Read2(BUFFER b, uint i) { return float2(ReadFloat(b,i),ReadFloat(b,i+1)); }
float4 Read4(BUFFER b, uint i) { return float4(ReadFloat(b,i),ReadFloat(b,i+1),ReadFloat(b,i+2),ReadFloat(b,i+3)); }
void Write2(BUFFER b, uint i, float2 v) { WriteFloat(b,i,v.x); WriteFloat(b,i+1,v.y); }
void Write4(BUFFER b, uint i, float4 v) { WriteFloat(b,i,v.x); WriteFloat(b,i+1,v.y); WriteFloat(b,i+2,v.z); WriteFloat(b,i+3,v.w); }
struct MpmSvd { float4 U; float4 V; float2 Stretch; };
MpmSvd MSvd(float4 f) {
    float a=f.x*f.x+f.z*f.z, b=f.x*f.y+f.z*f.w, c=f.y*f.y+f.w*f.w;
    // atan2(0,0) is undefined in HLSL; an isotropic stretch needs no rotation.
    float angle=(abs(b)+abs(a-c)>1.e-12) ? 0.5*atan2(2*b,a-c) : 0;
    float co=cos(angle), si=sin(angle);
    MpmSvd s;
    s.V=float4(co,-si,si,co);
    float2 first=MApply(f,float2(co,si)), second=MApply(f,float2(-si,co));
    s.Stretch=max(float2(length(first),length(second)),float2(1.e-6,1.e-6));
    float2 u=first/s.Stretch.x;
    s.U=s.Stretch.x<=1.e-6 ? MIdentity() : float4(u.x,-u.y,u.y,u.x);
    return s;
}
float2 MLame(float plasticVolume, DATA) {
    float hard=exp(clamp(PF(data,16)*(1-plasticVolume)*PF(data,14),-2.0,1.6));
    float e=PF(data,12)*hard, nu=PF(data,13);
    return float2(e/(2*(1+nu)), e*nu/((1+nu)*(1-2*nu)));
}
float4 MStress(float4 f, float4 c, float plasticVolume, DATA) {
    float2 lame=MLame(plasticVolume,data);
    MpmSvd s=MSvd(f);
    float4 r=MMultiply(s.U,MTranspose(s.V));
    float j=max(MDet(f),1.e-6);
    float4 snow=2*lame.x*MMultiply(f-r,MTranspose(f))+MIdentity()*(lame.y*j*(j-1));
    float2 strain=log(max(s.Stretch,float2(1.e-6,1.e-6)));
    float2 principal=2*lame.x*strain+lame.y*(strain.x+strain.y);
    float4 sand=MMultiply(MMultiply(s.U,float4(principal.x,0,0,principal.y)),MTranspose(s.U));
    return snow*PF(data,14)+sand*(1-PF(data,14))+(c+MTranspose(c))*PF(data,15);
}
struct MpmProjection { float4 Elastic; float Plastic; };
MpmProjection MProject(float4 trial, float plasticVolume, DATA) {
    MpmSvd s=MSvd(trial);
    float2 lame=MLame(plasticVolume,data);
    float2 snow=clamp(s.Stretch,float2(1-PF(data,17),1-PF(data,17)),float2(1+PF(data,18),1+PF(data,18)));
    float2 strain=log(max(s.Stretch,float2(1.e-6,1.e-6)));
    float tr=strain.x+strain.y;
    float2 dev=strain-0.5*tr;
    float norm=length(dev), si=sin(PF(data,19)*0.01745329252);
    float alpha=0.81649658093*2*si/(3-si);
    float plasticShear=norm+(lame.y+lame.x)/lame.x*tr*alpha;
    if(tr>=0) strain=float2(0,0);
    else if(plasticShear>0 && norm>1.e-8) strain-=plasticShear*dev/norm;
    float2 stretch=exp(PF(data,14)*log(snow)+(1-PF(data,14))*strain);
    MpmProjection p;
    p.Plastic=clamp(plasticVolume*max(MDet(trial),1.e-6)/max(stretch.x*stretch.y,1.e-6),0.4,4.0);
    p.Elastic=MMultiply(MMultiply(s.U,float4(stretch.x,0,0,stretch.y)),MTranspose(s.V));
    return p;
}
float MW(int i,float f) {
    if(i==0) return 0.5*(1.5-f)*(1.5-f);
    if(i==1) return 0.75-(f-1)*(f-1);
    return 0.5*(f-0.5)*(f-0.5);
}
float MDW(int i,float f) {
    if(i==0) return f-1.5;
    if(i==1) return -2*(f-1);
    return f-0.5;
}
float2 MContact(float2 position,float2 velocity,float padding,DATA) {
    if(PI(data,28)==0) return velocity;
    float2 center=float2(PF(data,21),PF(data,22)), capVelocity=float2(PF(data,23),PF(data,24));
    float2 closest=float2(center.x,clamp(position.y,center.y-PF(data,25),center.y+PF(data,25)));
    float2 offset=position-closest;
    float distance=length(offset);
    if(distance>=PF(data,26)+padding) return velocity;
    float2 normal=distance>1.e-6 ? offset/distance : float2(1,0);
    float2 relative=velocity-capVelocity;
    float incoming=dot(relative,normal);
    if(incoming>=0) return velocity;
    float2 tangent=relative-incoming*normal;
    float friction=max(0.0,1-PF(data,27)*(-incoming)/max(length(tangent),1.e-6));
    return capVelocity+tangent*friction;
}
KERNEL(SnowClear) {
    if(tid.x<(uint)(PI(constants,1)*PI(constants,2))) WriteUInt(b2,tid.x,0xffffffffu);
}
KERNEL(SnowPrepare) {
    uint id=tid.x;
    if(id>=(uint)PI(constants,0)) return;
    uint base=id*16;
    float4 stress=MStress(Read4(b0,base+4),Read4(b0,base+8),ReadFloat(b0,base+12),constants);
    Write4(b4,id*4,stress);
    float2 grid=(Read2(b0,base)-float2(PF(constants,4),PF(constants,5)))/PF(constants,8);
    int2 cell=int2(floor(grid-0.5));
    uint old=Exchange(b2,(uint)(cell.y*PI(constants,1)+cell.x),id);
    WriteUInt(b3,id,old);
}
KERNEL(SnowGrid) {
    uint id=tid.x;
    int width=PI(constants,1), height=PI(constants,2), count=PI(constants,0);
    if(id>=(uint)(width*height)) return;
    int x=(int)id%width, y=(int)id/width;
    float dx=PF(constants,8), inverse=1/dx;
    float2 origin=float2(PF(constants,4),PF(constants,5));
    float mass=0;
    float2 momentum=float2(0,0), force=float2(0,0);
    for(int by=max(0,y-2);by<=y;++by) for(int bx=max(0,x-2);bx<=x;++bx) {
        int visits=0;
        for(uint p=ReadUInt(b2,(uint)(by*width+bx));p<(uint)count && visits++<count;p=ReadUInt(b3,p)) {
            uint base=p*16;
            float2 f=(Read2(b0,base)-origin)*inverse-float2(bx,by);
            int i=x-bx, j=y-by;
            float wx=MW(i,f.x), wy=MW(j,f.y), w=wx*wy;
            float2 d=(float2(i,j)-f)*dx;
            float2 grad=PI(constants,3)==2 ? (4*inverse*inverse*w)*d : float2(MDW(i,f.x)*wy,MDW(j,f.y)*wx)*inverse;
            float4 affine=PI(constants,3)==0 ? float4(0,0,0,0) : Read4(b0,base+8);
            float4 stress=Read4(b4,p*4);
            float contribution=w*ReadFloat(b0,base+14);
            mass+=contribution;
            momentum+=contribution*(Read2(b0,base+2)+MApply(affine,d));
            force-=ReadFloat(b0,base+13)*MApply(stress,grad);
        }
    }
    float2 oldVelocity=float2(0,0), velocity=float2(0,0);
    if(mass>0) {
        oldVelocity=momentum/mass;
        velocity=oldVelocity+PF(constants,9)*(force/mass+float2(PF(constants,10),PF(constants,11)));
        float2 position=origin+float2(x,y)*dx;
        if(x<2 && velocity.x<0) velocity.x=0;
        if(y<2 && velocity.y<0) velocity.y=0;
        if(position.x>PF(constants,6)-2*dx && velocity.x>0) velocity.x=0;
        if(position.y>PF(constants,7)-2*dx && velocity.y>0) velocity.y=0;
        velocity=MContact(position,velocity,0.25*dx,constants);
        if(position.y<=PF(constants,29)) velocity=float2(0,0);
    }
    Write2(b1,id*6,velocity); Write2(b1,id*6+2,oldVelocity); WriteFloat(b1,id*6+4,mass);
}
KERNEL(SnowParticles) {
    uint id=tid.x;
    if(id>=(uint)PI(constants,0)) return;
    uint base=id*16;
    if(ReadFloat(b0,base+15)>0.5) return;
    int width=PI(constants,1);
    float dx=PF(constants,8), dt=PF(constants,9), inverse=1/dx;
    float2 origin=float2(PF(constants,4),PF(constants,5));
    float2 position=Read2(b0,base), previousVelocity=Read2(b0,base+2);
    float2 grid=(position-origin)*inverse;
    int2 cell=int2(floor(grid-0.5));
    float2 f=grid-float2(cell);
    float2 pic=float2(0,0), change=float2(0,0);
    float4 gradient=float4(0,0,0,0), affine=float4(0,0,0,0);
    for(int j=0;j<3;++j) for(int i=0;i<3;++i) {
        uint node=(uint)((cell.y+j)*width+cell.x+i)*6;
        float2 velocity=Read2(b1,node), oldVelocity=Read2(b1,node+2);
        float wx=MW(i,f.x), wy=MW(j,f.y), w=wx*wy;
        float2 d=(float2(i,j)-f)*dx;
        float2 grad=PI(constants,3)==2 ? (4*inverse*inverse*w)*d : float2(MDW(i,f.x)*wy,MDW(j,f.y)*wx)*inverse;
        float2 weighted=velocity*w;
        pic+=weighted; change+=w*(velocity-oldVelocity);
        gradient+=MOuter(velocity,grad);
        affine+=MOuter(weighted,d)*(4*inverse*inverse);
    }
    float2 velocity=PI(constants,3)==0 ? (1-PF(constants,20))*pic+PF(constants,20)*(previousVelocity+change) : pic;
    MpmProjection projected=MProject(MMultiply(MIdentity()+gradient*dt,Read4(b0,base+4)),ReadFloat(b0,base+12),constants);
    position+=dt*pic;
    float2 low=origin+1.5*dx, high=float2(PF(constants,6),PF(constants,7))-1.5*dx;
    if(position.x<low.x || position.x>high.x) velocity.x=0;
    if(position.y<low.y || position.y>high.y) velocity.y=0;
    position=clamp(position,low,high);
    velocity=MContact(position,velocity,0,constants);
    if(PI(constants,28)!=0) {
        float2 center=float2(PF(constants,21),PF(constants,22));
        float2 closest=float2(center.x,clamp(position.y,center.y-PF(constants,25),center.y+PF(constants,25)));
        float2 offset=position-closest;
        float distance=length(offset);
        if(distance<PF(constants,26)) position=closest+PF(constants,26)*(distance>1.e-6 ? offset/distance : float2(1,0));
    }
    Write2(b0,base,clamp(position,low,high)); Write2(b0,base+2,velocity);
    Write4(b0,base+4,projected.Elastic); Write4(b0,base+8,affine); WriteFloat(b0,base+12,projected.Plastic);
}
)MPM";
}
