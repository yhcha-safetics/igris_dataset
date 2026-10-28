// Double-precision vectors / quaternions and the frame conversions of docs/sim_protocol.md.
//   MuJoCo world: right-handed, Z up, +X forward, +Y left, quaternion (w, x, y, z)
//   native      : right-handed, Y up, Z back (XRoboToolkit / PICO), pose7 [x, y, z, qx, qy, qz, qw]
//   Unity       : left-handed, Y up, Z forward, X right
using System;

namespace IgrisPico.Core
{
    public struct V3
    {
        public double x, y, z;
        public V3(double x, double y, double z) { this.x = x; this.y = y; this.z = z; }
        public static readonly V3 Zero = new V3(0, 0, 0);

        public static V3 operator +(V3 a, V3 b) { return new V3(a.x + b.x, a.y + b.y, a.z + b.z); }
        public static V3 operator -(V3 a, V3 b) { return new V3(a.x - b.x, a.y - b.y, a.z - b.z); }
        public static V3 operator -(V3 a) { return new V3(-a.x, -a.y, -a.z); }
        public static V3 operator *(V3 a, double s) { return new V3(a.x * s, a.y * s, a.z * s); }
        public static V3 operator *(double s, V3 a) { return a * s; }
        public static double Dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        public static V3 Cross(V3 a, V3 b) { return new V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
        public static V3 Lerp(V3 a, V3 b, double t) { return a + (b - a) * t; }
        public double Norm { get { return Math.Sqrt(x * x + y * y + z * z); } }

        public V3 Normalized()
        {
            double n = Norm;
            return n > 1e-12 ? this * (1.0 / n) : Zero;
        }

        /// <summary>From a[off..off+2] (V3.Zero if a is too short).</summary>
        public static V3 From(double[] a, int off = 0)
        {
            return a != null && a.Length >= off + 3 ? new V3(a[off], a[off + 1], a[off + 2]) : Zero;
        }

        public override string ToString() { return string.Format("({0:F4}, {1:F4}, {2:F4})", x, y, z); }
    }

    /// <summary>Quaternion (w, x, y, z), Hamilton product, rotating vectors as q v q*.</summary>
    public struct Q4
    {
        public double w, x, y, z;
        public Q4(double w, double x, double y, double z) { this.w = w; this.x = x; this.y = y; this.z = z; }
        public static readonly Q4 Identity = new Q4(1, 0, 0, 0);

        public static Q4 operator *(Q4 a, Q4 b)
        {
            return new Q4(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
        }

        public Q4 Conj() { return new Q4(w, -x, -y, -z); }
        public static double Dot(Q4 a, Q4 b) { return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; }

        public Q4 Normalized()
        {
            double n = Math.Sqrt(w * w + x * x + y * y + z * z);
            return n > 1e-12 ? new Q4(w / n, x / n, y / n, z / n) : Identity;
        }

        public V3 Rotate(V3 v)
        {
            var u = new V3(x, y, z);
            V3 t = 2.0 * V3.Cross(u, v);
            return v + w * t + V3.Cross(u, t);
        }

        public static Q4 AxisAngle(V3 axis, double angle)
        {
            double s = Math.Sin(0.5 * angle);
            return new Q4(Math.Cos(0.5 * angle), axis.x * s, axis.y * s, axis.z * s);
        }

        public static Q4 Slerp(Q4 a, Q4 b, double t)
        {
            double d = Dot(a, b);
            if (d < 0) { b = new Q4(-b.w, -b.x, -b.y, -b.z); d = -d; }
            if (d > 0.9995)
                return new Q4(a.w + (b.w - a.w) * t, a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t).Normalized();
            double th = Math.Acos(Math.Min(1.0, d));
            double s0 = Math.Sin((1 - t) * th) / Math.Sin(th), s1 = Math.Sin(t * th) / Math.Sin(th);
            return new Q4(s0 * a.w + s1 * b.w, s0 * a.x + s1 * b.x, s0 * a.y + s1 * b.y, s0 * a.z + s1 * b.z);
        }

        /// <summary>From wxyz at a[off..off+3] (identity if a is too short).</summary>
        public static Q4 From(double[] a, int off = 0)
        {
            return a != null && a.Length >= off + 4 ? new Q4(a[off], a[off + 1], a[off + 2], a[off + 3]) : Identity;
        }

        public override string ToString() { return string.Format("({0:F4}, {1:F4}, {2:F4}, {3:F4})", w, x, y, z); }
    }

    public struct Pose
    {
        public V3 p;
        public Q4 q;
        public Pose(V3 p, Q4 q) { this.p = p; this.q = q; }
        public static readonly Pose Identity = new Pose(V3.Zero, Q4.Identity);
        public static Pose operator *(Pose a, Pose b) { return new Pose(a.p + a.q.Rotate(b.p), a.q * b.q); }
        public Pose Inverse() { Q4 c = q.Conj(); return new Pose(-c.Rotate(p), c); }
        public V3 Transform(V3 v) { return p + q.Rotate(v); }
    }

    /// <summary>A Unity-convention quaternion (x, y, z, w), for handing values to UnityEngine.Quaternion.</summary>
    public struct UQuat
    {
        public double x, y, z, w;
        public UQuat(double x, double y, double z, double w) { this.x = x; this.y = y; this.z = z; this.w = w; }
    }

    /// <summary>Frame conversions (reference: igris_sim/protocol.py, tested against it).</summary>
    public static class Conv
    {
        /// <summary>MuJoCo world position -> Unity position (x, y, z fields hold Unity x, y, z).</summary>
        public static V3 UnityFromMj(V3 p) { return new V3(-p.y, p.z, p.x); }

        /// <summary>MuJoCo quaternion (wxyz) -> Unity quaternion.</summary>
        public static UQuat UnityFromMj(Q4 q) { return new UQuat(q.y, -q.z, -q.x, q.w); }

        /// <summary>Unity tracking-space pose -> native pose7 [x, y, z, qx, qy, qz, qw].</summary>
        public static double[] NativeFromUnity(double px, double py, double pz, double qx, double qy, double qz, double qw)
        {
            return new[] { px, py, -pz, -qx, -qy, qz, qw };
        }

        /// <summary>Writes the native pose7 of a Unity pose into dst[off..off+6].</summary>
        public static void NativeFromUnity(double px, double py, double pz, double qx, double qy, double qz, double qw,
                                           double[] dst, int off = 0)
        {
            dst[off] = px; dst[off + 1] = py; dst[off + 2] = -pz;
            dst[off + 3] = -qx; dst[off + 4] = -qy; dst[off + 5] = qz; dst[off + 6] = qw;
        }

        /// <summary>Rotation about the fixed X, then Y, then Z axes (degrees) -> wxyz, as scipy "xyz".</summary>
        public static Q4 EulerXyzDeg(double rx, double ry, double rz)
        {
            const double d = Math.PI / 180.0;
            return (Q4.AxisAngle(new V3(0, 0, 1), rz * d) * Q4.AxisAngle(new V3(0, 1, 0), ry * d)
                    * Q4.AxisAngle(new V3(1, 0, 0), rx * d)).Normalized();
        }
    }
}
