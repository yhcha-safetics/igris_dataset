// OBJ / STL loading and conversion of MuJoCo-frame meshes to Unity vertex data (no Unity dependency, so it
// can run on a worker thread and be unit-tested).
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace IgrisPico.Core
{
    /// <summary>Indexed triangle mesh in the file's frame (MuJoCo axes for the sim assets).</summary>
    public sealed class MeshData
    {
        public readonly List<V3> Pos = new List<V3>();
        public List<double[]> Color;   // per-vertex rgb (OBJ "v x y z r g b"), or null
        public readonly List<int> Tri = new List<int>();
        public int TriangleCount { get { return Tri.Count / 3; } }
    }

    /// <summary>Vertex data ready for UnityEngine.Mesh (Unity axes, Unity winding).</summary>
    public sealed class UnityMeshData
    {
        public float[] Vertices, Normals, Colors;   // xyz / xyz / rgba per vertex; Colors may be null
        public int[] Indices;
        public int VertexCount { get { return Vertices.Length / 3; } }
    }

    public static class MeshIO
    {
        public static MeshData Load(string path, byte[] data)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            if (ext == ".obj") return ParseObj(Encoding.UTF8.GetString(data));
            if (ext == ".stl") return ParseStl(data);
            throw new NotSupportedException("mesh format " + ext + " (use .obj or .stl)");
        }

        public static MeshData ParseObj(string text)
        {
            var m = new MeshData();
            var colors = new List<double[]>();
            var face = new List<int>();
            var ci = CultureInfo.InvariantCulture;
            using (var reader = new StringReader(text))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    line = line.Trim();
                    if (line.Length < 2) continue;
                    if (line[0] == 'v' && line[1] == ' ')
                    {
                        string[] t = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                        m.Pos.Add(new V3(double.Parse(t[1], ci), double.Parse(t[2], ci), double.Parse(t[3], ci)));
                        colors.Add(t.Length >= 7 ? new[] { double.Parse(t[4], ci), double.Parse(t[5], ci), double.Parse(t[6], ci) } : null);
                    }
                    else if (line[0] == 'f' && line[1] == ' ')
                    {
                        string[] t = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                        face.Clear();
                        for (int k = 1; k < t.Length; k++)
                        {
                            int slash = t[k].IndexOf('/');
                            int idx = int.Parse(slash < 0 ? t[k] : t[k].Substring(0, slash), ci);
                            face.Add(idx < 0 ? m.Pos.Count + idx : idx - 1);   // OBJ: 1-based, negative = relative
                        }
                        for (int k = 1; k + 1 < face.Count; k++)   // fan-triangulate polygons
                        {
                            m.Tri.Add(face[0]);
                            m.Tri.Add(face[k]);
                            m.Tri.Add(face[k + 1]);
                        }
                    }
                }
            }
            foreach (int i in m.Tri)
                if (i < 0 || i >= m.Pos.Count) throw new FormatException("OBJ: face index out of range");
            if (colors.Count > 0 && colors.TrueForAll(c => c != null)) m.Color = colors;
            return m;
        }

        public static MeshData ParseStl(byte[] data)
        {
            if (data.Length >= 84)
            {
                uint n = BitConverter.ToUInt32(data, 80);
                if (84 + 50L * n == data.Length) return ParseBinaryStl(data, (int)n);
            }
            return ParseAsciiStl(Encoding.ASCII.GetString(data));
        }

        static MeshData ParseBinaryStl(byte[] data, int n)
        {
            var m = new MeshData();
            var weld = new Dictionary<V3Key, int>();
            for (int f = 0; f < n; f++)
            {
                int off = 84 + 50 * f + 12;  // skip the facet normal
                for (int k = 0; k < 3; k++, off += 12)
                {
                    var p = new V3(BitConverter.ToSingle(data, off), BitConverter.ToSingle(data, off + 4), BitConverter.ToSingle(data, off + 8));
                    m.Tri.Add(Weld(m, weld, p));
                }
            }
            return m;
        }

        static MeshData ParseAsciiStl(string text)
        {
            var m = new MeshData();
            var weld = new Dictionary<V3Key, int>();
            var ci = CultureInfo.InvariantCulture;
            using (var reader = new StringReader(text))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    line = line.Trim();
                    if (!line.StartsWith("vertex", StringComparison.OrdinalIgnoreCase)) continue;
                    string[] t = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    m.Tri.Add(Weld(m, weld, new V3(double.Parse(t[1], ci), double.Parse(t[2], ci), double.Parse(t[3], ci))));
                }
            }
            if (m.Tri.Count % 3 != 0) throw new FormatException("STL: vertex count is not a multiple of 3");
            return m;
        }

        struct V3Key : IEquatable<V3Key>
        {
            readonly float x, y, z;
            public V3Key(V3 p) { x = (float)p.x; y = (float)p.y; z = (float)p.z; }
            public bool Equals(V3Key o) { return x == o.x && y == o.y && z == o.z; }
            public override bool Equals(object o) { return o is V3Key && Equals((V3Key)o); }
            public override int GetHashCode() { return (x.GetHashCode() * 397 ^ y.GetHashCode()) * 397 ^ z.GetHashCode(); }
        }

        static int Weld(MeshData m, Dictionary<V3Key, int> weld, V3 p)
        {
            var key = new V3Key(p);
            int i;
            if (!weld.TryGetValue(key, out i))
            {
                i = m.Pos.Count;
                m.Pos.Add(p);
                weld[key] = i;
            }
            return i;
        }

        /// <summary>
        /// MuJoCo-frame mesh (scaled per axis) -> Unity vertex data: axes converted with Conv.UnityFromMj, triangle
        /// winding reversed (the axis change is a reflection), normals smoothed between faces closer than creaseDeg.
        /// </summary>
        public static UnityMeshData ToUnity(MeshData m, V3 scale, double creaseDeg = 40.0)
        {
            int nv = m.Pos.Count, nf = m.TriangleCount;
            var pu = new V3[nv];
            for (int i = 0; i < nv; i++)
            {
                V3 p = m.Pos[i];
                pu[i] = Conv.UnityFromMj(new V3(p.x * scale.x, p.y * scale.y, p.z * scale.z));
            }
            var tri = new int[nf * 3];
            var fn = new V3[nf];      // area-weighted face normals (outward for outward-facing input)
            var fu = new V3[nf];      // unit face normals
            for (int f = 0; f < nf; f++)
            {
                int a = m.Tri[3 * f], b = m.Tri[3 * f + 1], c = m.Tri[3 * f + 2];
                tri[3 * f] = a; tri[3 * f + 1] = c; tri[3 * f + 2] = b;
                fn[f] = V3.Cross(pu[c] - pu[a], pu[b] - pu[a]);
                fu[f] = fn[f].Normalized();
            }
            // faces around each vertex
            var count = new int[nv + 1];
            for (int k = 0; k < tri.Length; k++) count[tri[k] + 1]++;
            for (int i = 0; i < nv; i++) count[i + 1] += count[i];
            var incident = new int[tri.Length];
            var fill = (int[])count.Clone();
            for (int k = 0; k < tri.Length; k++) incident[fill[tri[k]]++] = k;   // corner index (3 f + j)

            double cosCrease = Math.Cos(creaseDeg * Math.PI / 180.0);
            var outPos = new List<V3>(nv);
            var outNrm = new List<V3>(nv);
            var outSrc = new List<int>(nv);
            var cornerVertex = new int[tri.Length];
            var seeds = new List<V3>();
            var sums = new List<V3>();
            var members = new List<int>();
            for (int v = 0; v < nv; v++)
            {
                seeds.Clear(); sums.Clear(); members.Clear();
                for (int s = count[v]; s < count[v + 1]; s++)
                {
                    int corner = incident[s], f = corner / 3;
                    int cl = -1;
                    for (int c = 0; c < seeds.Count; c++)
                        if (V3.Dot(seeds[c], fu[f]) >= cosCrease) { cl = c; break; }
                    if (cl < 0) { cl = seeds.Count; seeds.Add(fu[f]); sums.Add(V3.Zero); }
                    sums[cl] = sums[cl] + fn[f];
                    members.Add(cl);
                }
                int baseIndex = outPos.Count;
                for (int c = 0; c < seeds.Count; c++)
                {
                    V3 n = sums[c].Normalized();
                    outPos.Add(pu[v]);
                    outNrm.Add(n.Norm > 0 ? n : (seeds[c].Norm > 0 ? seeds[c] : new V3(0, 1, 0)));
                    outSrc.Add(v);
                }
                int mi = 0;
                for (int s = count[v]; s < count[v + 1]; s++) cornerVertex[incident[s]] = baseIndex + members[mi++];
            }

            var r = new UnityMeshData
            {
                Vertices = new float[outPos.Count * 3],
                Normals = new float[outPos.Count * 3],
                Indices = cornerVertex,
            };
            for (int i = 0; i < outPos.Count; i++)
            {
                r.Vertices[3 * i] = (float)outPos[i].x; r.Vertices[3 * i + 1] = (float)outPos[i].y; r.Vertices[3 * i + 2] = (float)outPos[i].z;
                r.Normals[3 * i] = (float)outNrm[i].x; r.Normals[3 * i + 1] = (float)outNrm[i].y; r.Normals[3 * i + 2] = (float)outNrm[i].z;
            }
            if (m.Color != null)
            {
                r.Colors = new float[outPos.Count * 4];
                for (int i = 0; i < outPos.Count; i++)
                {
                    double[] c = m.Color[outSrc[i]];
                    r.Colors[4 * i] = (float)c[0]; r.Colors[4 * i + 1] = (float)c[1]; r.Colors[4 * i + 2] = (float)c[2];
                    r.Colors[4 * i + 3] = 1f;
                }
            }
            return r;
        }
    }
}
