// Materials, primitives, meshes and text built at runtime (the app has no authored assets besides shaders).
using System.Collections.Generic;
using IgrisPico.Core;
using UnityEngine;
using UnityEngine.Rendering;

namespace IgrisPico
{
    public static class Visuals
    {
        static Shader lit, litTransparent, unlit, text;
        static Font font;
        static readonly List<Material> textMaterials = new List<Material>();

        static Shader Find(string name, string resource)
        {
            Shader s = Shader.Find(name);
            if (s == null) s = Resources.Load<Shader>(resource);
            if (s == null) Debug.LogError("[IGRIS] shader missing: " + name);
            return s;
        }

        public static Material Lit(Color c)
        {
            if (lit == null)
            {
                lit = Find("IgrisPico/Lit", "IgrisLit");
                litTransparent = Find("IgrisPico/LitTransparent", "IgrisLitTransparent");
            }
            return new Material(c.a < 0.99f ? litTransparent : lit) { color = c };
        }

        public static Material Lit(double[] rgba)
        {
            return Lit(ToColor(rgba));
        }

        /// <summary>Unlit colour; onTop: drawn over the scene (HUD).</summary>
        public static Material Unlit(Color c, bool onTop = false)
        {
            if (unlit == null) unlit = Find("IgrisPico/Unlit", "IgrisUnlit");
            var m = new Material(unlit) { color = c };
            m.SetFloat("_ZTest", (float)(onTop ? CompareFunction.Always : CompareFunction.LessEqual));
            return m;
        }

        public static Color ToColor(double[] rgba)
        {
            if (rgba == null || rgba.Length < 3) return new Color(0.7f, 0.7f, 0.7f, 1f);
            return new Color((float)rgba[0], (float)rgba[1], (float)rgba[2], rgba.Length > 3 ? (float)rgba[3] : 1f);
        }

        public static Vector3 ToUnity(V3 mjPos)
        {
            V3 u = Conv.UnityFromMj(mjPos);
            return new Vector3((float)u.x, (float)u.y, (float)u.z);
        }

        public static Quaternion ToUnity(Q4 mjQuat)
        {
            UQuat u = Conv.UnityFromMj(mjQuat);
            return new Quaternion((float)u.x, (float)u.y, (float)u.z, (float)u.w);
        }

        public static void SetLocal(Transform t, Core.Pose mjPose)
        {
            t.localPosition = ToUnity(mjPose.p);
            t.localRotation = ToUnity(mjPose.q);
        }

        /// <summary>A Unity primitive without collider (the PC simulates physics).</summary>
        public static GameObject Primitive(PrimitiveType type, Transform parent, Material m, string name = null)
        {
            GameObject go = GameObject.CreatePrimitive(type);
            Object.Destroy(go.GetComponent<Collider>());
            if (name != null) go.name = name;
            go.transform.SetParent(parent, false);
            var r = go.GetComponent<MeshRenderer>();
            r.sharedMaterial = m;
            r.shadowCastingMode = ShadowCastingMode.Off;
            r.receiveShadows = false;
            return go;
        }

        public static PrimitiveType PrimitiveFor(string shape)
        {
            switch (shape)
            {
                case "sphere": return PrimitiveType.Sphere;
                case "cylinder": return PrimitiveType.Cylinder;
                case "capsule": return PrimitiveType.Capsule;
                default: return PrimitiveType.Cube;
            }
        }

        public static Mesh BuildMesh(UnityMeshData d, string name)
        {
            int n = d.VertexCount;
            var v = new Vector3[n];
            var nr = new Vector3[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = new Vector3(d.Vertices[3 * i], d.Vertices[3 * i + 1], d.Vertices[3 * i + 2]);
                nr[i] = new Vector3(d.Normals[3 * i], d.Normals[3 * i + 1], d.Normals[3 * i + 2]);
            }
            var mesh = new Mesh { name = name, indexFormat = n > 65000 ? IndexFormat.UInt32 : IndexFormat.UInt16 };
            mesh.vertices = v;
            mesh.normals = nr;
            if (d.Colors != null)
            {
                var c = new Color[n];
                for (int i = 0; i < n; i++) c[i] = new Color(d.Colors[4 * i], d.Colors[4 * i + 1], d.Colors[4 * i + 2], d.Colors[4 * i + 3]);
                mesh.colors = c;
            }
            mesh.SetTriangles(d.Indices, 0);
            mesh.RecalculateBounds();
            return mesh;
        }

        public static GameObject MeshObject(Mesh mesh, Transform parent, Material m, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var r = go.AddComponent<MeshRenderer>();
            r.sharedMaterial = m;
            r.shadowCastingMode = ShadowCastingMode.Off;
            r.receiveShadows = false;
            return go;
        }

        /// <summary>A world-space TextMesh (charSize m per font unit) drawn over the scene.</summary>
        public static TextMesh Text(Transform parent, string name, float charSize, TextAnchor anchor, Color color)
        {
            if (font == null)
            {
                font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
                if (text == null) text = Find("IgrisPico/Text", "IgrisText");
                Font.textureRebuilt += f =>
                {
                    if (f != font) return;
                    foreach (Material m in textMaterials) if (m != null) m.mainTexture = font.material.mainTexture;
                };
            }
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var tm = go.AddComponent<TextMesh>();
            tm.font = font;
            tm.fontSize = 64;
            tm.characterSize = charSize;
            tm.anchor = anchor;
            tm.alignment = anchor == TextAnchor.UpperCenter || anchor == TextAnchor.MiddleCenter ? TextAlignment.Center : TextAlignment.Left;
            tm.color = color;
            var mat = new Material(text) { mainTexture = font.material.mainTexture };
            textMaterials.Add(mat);
            go.GetComponent<MeshRenderer>().sharedMaterial = mat;
            return tm;
        }

        /// <summary>Floor grid texture: light tiles with darker 1 m lines.</summary>
        public static Texture2D GridTexture(Color tile, Color line)
        {
            const int n = 128;
            var t = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Repeat, filterMode = FilterMode.Trilinear, anisoLevel = 4 };
            var px = new Color[n * n];
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                    px[y * n + x] = x < 2 || y < 2 ? line : tile;
            t.SetPixels(px);
            t.Apply(true);
            return t;
        }
    }
}
