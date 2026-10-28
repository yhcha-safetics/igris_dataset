// scene.json (igris_sim/scene.py) as seen by the app. Positions / rotations are MuJoCo-frame; sizes are full
// extents (box [sx, sy, sz], sphere [r], cylinder / capsule [r, height]).
using System;
using System.Collections.Generic;

namespace IgrisPico.Core
{
    public sealed class ScenePart
    {
        public string Shape;      // box | sphere | cylinder | capsule | mesh
        public double[] Size;
        public Pose Local;        // in the object (body) frame
        public double[] Rgba;
        public string File, Render;
        public V3 Scale = new V3(1, 1, 1);
        public bool Collide = true;
    }

    public sealed class SceneObject
    {
        public string Name;
        public Pose Pose;         // initial body pose in the world
        public bool Dynamic, Graspable;
        public readonly List<ScenePart> Parts = new List<ScenePart>();
    }

    public sealed class SceneSpec
    {
        public string Name, Description;
        public V3 Spawn;          // x, y (z = 0)
        public double SpawnYawDeg;
        public bool Floor = true;
        public double[] FloorRgba;  // null: default floor
        public readonly List<SceneObject> Objects = new List<SceneObject>();

        static readonly double[] DefaultRgba = { 0.7, 0.7, 0.7, 1.0 };

        public static SceneSpec FromJson(Dictionary<string, object> spec)
        {
            if (Json.Str(Json.Get(spec, "format")) != "igris_scene") throw new FormatException("not an igris_scene");
            var s = new SceneSpec
            {
                Name = Json.Str(Json.Get(spec, "name"), ""),
                Description = Json.Str(Json.Get(spec, "description"), ""),
            };
            var spawn = Json.Obj(Json.Get(spec, "spawn"));
            double[] sp = Json.Nums(Json.Get(spawn, "pos"));
            s.Spawn = sp != null && sp.Length >= 2 ? new V3(sp[0], sp[1], 0) : V3.Zero;
            s.SpawnYawDeg = Json.Num(Json.Get(spawn, "yaw_deg"));
            object floor = Json.Get(spec, "floor");
            if (floor is bool && !(bool)floor) s.Floor = false;
            else s.FloorRgba = Json.Nums(Json.Get(Json.Obj(floor), "rgba"));

            foreach (var o in Json.Arr(Json.Get(spec, "objects")) ?? new List<object>())
            {
                var od = Json.Obj(o);
                if (od == null) continue;
                bool dyn = Json.Bool(Json.Get(od, "dynamic"));
                var obj = new SceneObject
                {
                    Name = Json.Str(Json.Get(od, "name"), ""),
                    Pose = new Pose(V3.From(Json.Nums(Json.Get(od, "pos"))), Rot(Json.Get(od, "rot"))),
                    Dynamic = dyn,
                    Graspable = dyn && Json.Bool(Json.Get(od, "graspable"), dyn),
                };
                var geoms = Json.Arr(Json.Get(od, "geoms"));
                if (geoms != null)
                    foreach (var g in geoms) obj.Parts.Add(Part(Json.Obj(g), od));
                else
                    obj.Parts.Add(Part(od, od, ownPose: false));
                s.Objects.Add(obj);
            }
            return s;
        }

        /// <summary>One geom part; the object's rgba / collide are the defaults (scene.py _parts).</summary>
        static ScenePart Part(Dictionary<string, object> g, Dictionary<string, object> obj, bool ownPose = true)
        {
            var p = new ScenePart
            {
                Shape = Json.Str(Json.Get(g, "shape"), ""),
                Size = Json.Nums(Json.Get(g, "size")) ?? new double[0],
                Local = ownPose ? new Pose(V3.From(Json.Nums(Json.Get(g, "pos"))), Rot(Json.Get(g, "rot"))) : Pose.Identity,
                Rgba = Json.Nums(Json.Get(g, "rgba")) ?? Json.Nums(Json.Get(obj, "rgba")) ?? DefaultRgba,
                File = Json.Str(Json.Get(g, "file")),
                Render = Json.Str(Json.Get(g, "render")),
                Collide = Json.Bool(Json.Get(g, "collide"), Json.Bool(Json.Get(obj, "collide"), true)),
            };
            double[] sc = Json.Nums(Json.Get(g, "scale"));
            if (sc != null && sc.Length >= 3) p.Scale = new V3(sc[0], sc[1], sc[2]);
            return p;
        }

        /// <summary>Scene rotation -> wxyz (scene.py rot_quat).</summary>
        public static Q4 Rot(object rot)
        {
            var r = Json.Obj(rot);
            if (r == null) return Q4.Identity;
            double[] q = Json.Nums(Json.Get(r, "quat"));
            if (q != null && q.Length == 4) return Q4.From(q).Normalized();
            double[] e = Json.Nums(Json.Get(r, "euler_deg"));
            return e != null && e.Length == 3 ? Conv.EulerXyzDeg(e[0], e[1], e[2]) : Q4.Identity;
        }

        /// <summary>
        /// localScale of the Unity primitive (Cube, Sphere, Cylinder, Capsule) for a part, in Unity axes;
        /// the primitive's axis (Unity local Y) is the MuJoCo local Z once the pose is converted.
        /// </summary>
        public static V3 UnityPrimitiveScale(string shape, double[] size)
        {
            switch (shape)
            {
                case "box": return new V3(size[1], size[2], size[0]);
                case "sphere": return new V3(2 * size[0], 2 * size[0], 2 * size[0]);
                case "cylinder": return new V3(2 * size[0], size[1] / 2, 2 * size[0]);
                case "capsule": return new V3(2 * size[0], (size[1] + 2 * size[0]) / 2, 2 * size[0]);
                default: throw new ArgumentException("not a primitive: " + shape);
            }
        }

        /// <summary>The Unity primitive for MuJoCo half sizes (robot.json geoms).</summary>
        public static V3 UnityPrimitiveScaleFromHalf(string shape, double[] half)
        {
            switch (shape)
            {
                case "box": return UnityPrimitiveScale(shape, new[] { 2 * half[0], 2 * half[1], 2 * half[2] });
                case "sphere": return UnityPrimitiveScale(shape, new[] { half[0] });
                default: return UnityPrimitiveScale(shape, new[] { half[0], 2 * half[1] });
            }
        }
    }
}
