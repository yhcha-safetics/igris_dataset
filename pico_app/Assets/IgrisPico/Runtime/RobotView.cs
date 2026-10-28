// IGRIS-C drawn from the server's robot description: one transform per body, posed every frame from the
// interpolated qpos with the description's forward kinematics.
using System.Collections;
using System.Collections.Generic;
using IgrisPico.Core;
using UnityEngine;

namespace IgrisPico
{
    public sealed class RobotView
    {
        public RobotModel Model { get; private set; }
        public bool Ready { get; private set; }
        readonly Transform root;
        Transform[] bodies;
        Core.Pose[] world;
        readonly List<Renderer> headRenderers = new List<Renderer>();
        readonly List<Renderer> renderers = new List<Renderer>();
        Material solid, ghost, solidDark, ghostDark;
        bool ghosted, headVisible = true;

        public const string HeadBody = "Link_Neck_Yaw";   // hidden in the embodied view (the camera is inside it)

        public RobotView(Transform parent)
        {
            root = new GameObject("Robot").transform;
            root.SetParent(parent, false);
        }

        public void Clear()
        {
            Ready = false;
            for (int i = root.childCount - 1; i >= 0; i--) Object.Destroy(root.GetChild(i).gameObject);
            headRenderers.Clear();
            renderers.Clear();
        }

        public IEnumerator Build(RobotModel model, string baseUrl, AssetCache cache)
        {
            Clear();
            Model = model;
            solid = Visuals.Lit(new Color(0.80f, 0.82f, 0.85f, 1f));
            solidDark = Visuals.Lit(new Color(0.25f, 0.27f, 0.30f, 1f));
            ghost = Visuals.Lit(new Color(0.75f, 0.85f, 1.0f, 0.35f));
            ghostDark = Visuals.Lit(new Color(0.3f, 0.4f, 0.5f, 0.35f));
            int n = model.Bodies.Count;
            bodies = new Transform[n];
            world = new Core.Pose[n];
            for (int k = 0; k < n; k++)
            {
                bodies[k] = new GameObject(model.Bodies[k].Name).transform;
                bodies[k].SetParent(root, false);
            }
            int head = model.BodyIndex(HeadBody);
            foreach (RobotGeom g in model.Geoms)
            {
                GameObject go = null;
                bool dark = g.Name.Contains("Hand") || g.Name.Contains("Ankle_Roll");
                if (g.Type == "mesh")
                {
                    Mesh mesh = null;
                    yield return cache.GetMesh(baseUrl + g.Mesh, new V3(1, 1, 1), m => mesh = m);
                    if (mesh == null) continue;
                    go = Visuals.MeshObject(mesh, bodies[g.Body], dark ? solidDark : solid, g.Name);
                }
                else if (g.Size != null)
                {
                    go = Visuals.Primitive(Visuals.PrimitiveFor(g.Type), bodies[g.Body], dark ? solidDark : solid, g.Name);
                    V3 s = SceneSpec.UnityPrimitiveScaleFromHalf(g.Type, g.Size);
                    go.transform.localScale = new Vector3((float)s.x, (float)s.y, (float)s.z);
                }
                if (go == null) continue;
                Visuals.SetLocal(go.transform, g.Local);
                var r = go.GetComponent<Renderer>();
                renderers.Add(r);
                if (head >= 0 && model.Descends(g.Body, head)) headRenderers.Add(r);
            }
            Ready = true;
            ApplyStyle();
        }

        public void Apply(double[] qpos)
        {
            if (!Ready || qpos == null) return;
            Model.ForwardKinematics(qpos, world);
            for (int k = 0; k < bodies.Length; k++)
            {
                bodies[k].localPosition = Visuals.ToUnity(world[k].p);
                bodies[k].localRotation = Visuals.ToUnity(world[k].q);
            }
        }

        /// <summary>World pose of a body (Unity), e.g. a hand for grasp highlighting.</summary>
        public Transform Body(string name)
        {
            int k = Model == null ? -1 : Model.BodyIndex(name);
            return k >= 0 && bodies != null ? bodies[k] : null;
        }

        public void SetStyle(bool ghostLook, bool showHead)
        {
            if (ghostLook == ghosted && showHead == headVisible) return;
            ghosted = ghostLook;
            headVisible = showHead;
            ApplyStyle();
        }

        void ApplyStyle()
        {
            foreach (Renderer r in renderers)
            {
                bool dark = r.name.Contains("Hand") || r.name.Contains("Ankle_Roll");
                r.sharedMaterial = ghosted ? (dark ? ghostDark : ghost) : (dark ? solidDark : solid);
            }
            foreach (Renderer r in headRenderers) r.enabled = headVisible;
        }
    }
}
