// The robot description published by the server (igris_sim/robot_export.py) and its forward kinematics.
using System;
using System.Collections.Generic;

namespace IgrisPico.Core
{
    public sealed class RobotBody
    {
        public string Name;
        public int Parent;            // index into Bodies, -1 = world
        public Pose Local;            // pose in the parent frame at zero joint angle
        public bool Free;             // root with a free joint: pose = qpos[0:7]
        public int Qpos = -1;         // hinge: index into qpos
        public V3 Axis, JointPos;
        public double Ref;
    }

    public sealed class RobotGeom
    {
        public string Name, Type, Mesh;  // Type: mesh | box | sphere | capsule | cylinder
        public int Body;
        public Pose Local;
        public double[] Size;            // MuJoCo half sizes (primitives)
        public double[] Rgba;
    }

    public sealed class RobotModel
    {
        public readonly List<RobotBody> Bodies = new List<RobotBody>();
        public readonly List<RobotGeom> Geoms = new List<RobotGeom>();
        public readonly List<string> JointNames = new List<string>();
        public int Nq;

        public static RobotModel Parse(string json)
        {
            var d = Json.Obj(Json.Parse(json));
            if (Json.Str(Json.Get(d, "format")) != "igris_robot")
                throw new FormatException("not an igris_robot description");
            var r = new RobotModel { Nq = (int)Json.Num(Json.Get(d, "nq")) };
            foreach (var n in Json.Arr(Json.Get(d, "joint_names")) ?? new List<object>())
                r.JointNames.Add(Json.Str(n, ""));
            foreach (var o in Json.Arr(Json.Get(d, "bodies")) ?? new List<object>())
            {
                var b = Json.Obj(o);
                var body = new RobotBody
                {
                    Name = Json.Str(Json.Get(b, "name"), ""),
                    Parent = (int)Json.Num(Json.Get(b, "parent"), -1),
                    Local = new Pose(V3.From(Json.Nums(Json.Get(b, "pos"))), Q4.From(Json.Nums(Json.Get(b, "quat"))).Normalized()),
                };
                if (body.Parent >= r.Bodies.Count) throw new FormatException("body " + body.Name + ": parent listed after child");
                var j = Json.Obj(Json.Get(b, "joint"));
                string jt = Json.Str(Json.Get(j, "type"));
                if (jt == "free") body.Free = true;
                else if (jt == "hinge")
                {
                    body.Qpos = (int)Json.Num(Json.Get(j, "qpos"));
                    body.Axis = V3.From(Json.Nums(Json.Get(j, "axis"))).Normalized();
                    body.JointPos = V3.From(Json.Nums(Json.Get(j, "pos")));
                    body.Ref = Json.Num(Json.Get(j, "ref"));
                }
                else if (jt != null) throw new FormatException("body " + body.Name + ": unsupported joint " + jt);
                r.Bodies.Add(body);
            }
            foreach (var o in Json.Arr(Json.Get(d, "geoms")) ?? new List<object>())
            {
                var g = Json.Obj(o);
                r.Geoms.Add(new RobotGeom
                {
                    Name = Json.Str(Json.Get(g, "name"), ""),
                    Type = Json.Str(Json.Get(g, "type"), ""),
                    Mesh = Json.Str(Json.Get(g, "mesh")),
                    Body = (int)Json.Num(Json.Get(g, "body")),
                    Local = new Pose(V3.From(Json.Nums(Json.Get(g, "pos"))), Q4.From(Json.Nums(Json.Get(g, "quat"))).Normalized()),
                    Size = Json.Nums(Json.Get(g, "size")),
                    Rgba = Json.Nums(Json.Get(g, "rgba")) ?? new[] { 0.7, 0.7, 0.7, 1.0 },
                });
            }
            return r;
        }

        /// <summary>World poses (MuJoCo frame) of all bodies for qpos [root pos, root quat wxyz, joints].</summary>
        public void ForwardKinematics(double[] qpos, Pose[] world)
        {
            for (int k = 0; k < Bodies.Count; k++)
            {
                RobotBody b = Bodies[k];
                if (b.Free)
                {
                    world[k] = new Pose(V3.From(qpos, 0), Q4.From(qpos, 3).Normalized());
                    continue;
                }
                Pose x = (b.Parent >= 0 ? world[b.Parent] : Pose.Identity) * b.Local;
                if (b.Qpos >= 0 && b.Qpos < qpos.Length)
                {
                    V3 anchor = x.Transform(b.JointPos);
                    x.q = (x.q * Q4.AxisAngle(b.Axis, qpos[b.Qpos] - b.Ref)).Normalized();
                    x.p = anchor - x.q.Rotate(b.JointPos);
                }
                world[k] = x;
            }
        }

        public int BodyIndex(string name)
        {
            for (int k = 0; k < Bodies.Count; k++)
                if (Bodies[k].Name == name) return k;
            return -1;
        }

        /// <summary>True if body k is `ancestor` or one of its descendants.</summary>
        public bool Descends(int k, int ancestor)
        {
            for (; k >= 0; k = Bodies[k].Parent)
                if (k == ancestor) return true;
            return false;
        }
    }
}
