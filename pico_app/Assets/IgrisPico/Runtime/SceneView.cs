// The simulation scene rebuilt from the scene message: floor, fixed objects, and the dynamic objects that
// follow state.objects (in the order of scene.dynamic).
using System.Collections;
using System.Collections.Generic;
using System.IO;
using IgrisPico.Core;
using UnityEngine;

namespace IgrisPico
{
    public sealed class SceneView
    {
        public string Name { get; private set; }
        readonly Transform root;
        Transform[] dynamicObjects = new Transform[0];
        readonly Dictionary<string, Renderer[]> renderers = new Dictionary<string, Renderer[]>();
        readonly Dictionary<string, Color> emission = new Dictionary<string, Color>();
        Texture2D grid;

        public SceneView(Transform parent)
        {
            root = new GameObject("Scene").transform;
            root.SetParent(parent, false);
        }

        public void Clear()
        {
            for (int i = root.childCount - 1; i >= 0; i--) Object.Destroy(root.GetChild(i).gameObject);
            dynamicObjects = new Transform[0];
            renderers.Clear();
            emission.Clear();
            Name = null;
        }

        /// <summary>Builds the scene; meshes come from assetBase + msg.Folder + "/" + file.</summary>
        public IEnumerator Build(SceneMessage msg, string assetBase, AssetCache cache)
        {
            Clear();
            Name = msg.Name;
            SceneSpec spec = msg.Spec;
            if (spec.Floor) BuildFloor(spec.FloorRgba);
            var byName = new Dictionary<string, Transform>();
            foreach (SceneObject o in spec.Objects)
            {
                var t = new GameObject(o.Name).transform;
                t.SetParent(root, false);
                Visuals.SetLocal(t, o.Pose);
                byName[o.Name] = t;
                var rs = new List<Renderer>();
                for (int k = 0; k < o.Parts.Count; k++)
                {
                    ScenePart p = o.Parts[k];
                    if (p.Rgba.Length > 3 && p.Rgba[3] <= 0.001) continue;  // collider only (e.g. scanned walls)
                    GameObject go = null;
                    if (p.Shape == "mesh")
                    {
                        string file = Renderable(p.Render) ? p.Render : p.File;
                        if (string.IsNullOrEmpty(assetBase) || msg.Folder == null || string.IsNullOrEmpty(file)) continue;
                        Mesh mesh = null;
                        yield return cache.GetMesh(assetBase + msg.Folder + "/" + file, p.Scale, m => mesh = m);
                        if (mesh == null) continue;
                        // vertex-coloured scans keep their colours (white tint)
                        go = Visuals.MeshObject(mesh, t, Visuals.Lit(mesh.colors.Length > 0 ? new Color(1, 1, 1, (float)p.Rgba[3]) : Visuals.ToColor(p.Rgba)), o.Name + "_" + k);
                    }
                    else
                    {
                        go = Visuals.Primitive(Visuals.PrimitiveFor(p.Shape), t, Visuals.Lit(p.Rgba), o.Name + "_" + k);
                        V3 s = SceneSpec.UnityPrimitiveScale(p.Shape, p.Size);
                        go.transform.localScale = new Vector3((float)s.x, (float)s.y, (float)s.z);
                    }
                    Visuals.SetLocal(go.transform, p.Local);
                    rs.Add(go.GetComponent<Renderer>());
                }
                renderers[o.Name] = rs.ToArray();
            }
            dynamicObjects = new Transform[msg.Dynamic.Count];
            for (int i = 0; i < msg.Dynamic.Count; i++)
            {
                Transform t;
                dynamicObjects[i] = byName.TryGetValue(msg.Dynamic[i], out t) ? t : null;
            }
        }

        static bool Renderable(string file)
        {
            if (string.IsNullOrEmpty(file)) return false;
            string ext = Path.GetExtension(file).ToLowerInvariant();
            return ext == ".obj" || ext == ".stl";
        }

        void BuildFloor(double[] rgba)
        {
            if (grid == null) grid = Visuals.GridTexture(new Color(0.62f, 0.64f, 0.66f), new Color(0.42f, 0.44f, 0.47f));
            Color tint = rgba != null ? Visuals.ToColor(rgba) : Color.white;
            Material m = Visuals.Lit(new Color(tint.r, tint.g, tint.b, 1f));
            m.mainTexture = grid;
            const float size = 40f;   // metres; one texture tile per metre
            m.mainTextureScale = new Vector2(size, size);
            GameObject floor = Visuals.Primitive(PrimitiveType.Quad, root, m, "Floor");
            floor.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            floor.transform.localScale = new Vector3(size, size, 1f);
            floor.transform.localPosition = new Vector3(0f, -0.001f, 0f);
        }

        /// <summary>Poses (MuJoCo world) of the dynamic objects, in scene.dynamic order.</summary>
        public void Apply(V3[] pos, Q4[] quat, int count)
        {
            int n = Mathf.Min(count, dynamicObjects.Length);
            for (int i = 0; i < n; i++)
            {
                Transform t = dynamicObjects[i];
                if (t == null) continue;
                t.localPosition = Visuals.ToUnity(pos[i]);
                t.localRotation = Visuals.ToUnity(quat[i]);
            }
        }

        public int DynamicCount { get { return dynamicObjects.Length; } }

        /// <summary>Tints the grasped objects (null names are ignored).</summary>
        public void Highlight(string left, string right)
        {
            foreach (KeyValuePair<string, Renderer[]> kv in renderers)
            {
                Color want = kv.Key == left || kv.Key == right ? new Color(0.25f, 0.2f, 0.05f) : Color.black;
                Color have;
                if (emission.TryGetValue(kv.Key, out have) && have == want) continue;
                emission[kv.Key] = want;
                foreach (Renderer r in kv.Value) if (r != null) r.material.SetColor("_Emission", want);
            }
        }
    }
}
