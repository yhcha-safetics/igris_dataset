// Messages of docs/sim_protocol.md: tracking / cmd from the app, scene / state / ack / error / scenes from the server.
using System;
using System.Collections.Generic;

namespace IgrisPico.Core
{
    public static class Protocol
    {
        public const int Version = 1;
        public const int DefaultPort = 9870;
        public static readonly string[] Commands = { "hello", "list_scenes", "load_scene", "set_mode", "reset", "recalibrate", "record", "marker" };

        /// <summary>A cmd message; args are key, value pairs (string, bool or double values).</summary>
        public static string Cmd(string cmd, params object[] args)
        {
            var w = new JsonWriter().BeginObject().Key("v").Value(Version).Key("type").Value("cmd").Key("cmd").Value(cmd);
            for (int i = 0; i + 1 < args.Length; i += 2)
            {
                w.Key((string)args[i]);
                object v = args[i + 1];
                if (v is bool) w.Value((bool)v);
                else if (v is double) w.Value((double)v);
                else if (v is int) w.Value((int)v);
                else w.Value(v == null ? null : v.ToString());
            }
            return w.EndObject().ToString();
        }

        public static string TypeOf(Dictionary<string, object> msg) { return Json.Str(Json.Get(msg, "type"), ""); }
    }

    /// <summary>One tracking message. Poses are native pose7 [x, y, z, qx, qy, qz, qw] in the tracking space.</summary>
    public sealed class TrackingFrame
    {
        public const int BodyJoints = 24;
        public long TNs;
        public readonly double[] Head = new double[7], Left = new double[7], Right = new double[7];
        public bool A, B, X, Y, LeftMenu, RightMenu;
        public double LeftTrigger, RightTrigger, LeftGrip, RightGrip;
        public double LX, LY, RX, RY;
        public double[] Body;     // 24 x 7 (PICO BodyTrackerRole order), or null when there is no new sample
        public long[] BodyTs;     // 24, with Body
        public readonly List<string> TrackerSn = new List<string>();
        public readonly List<double[]> TrackerPose = new List<double[]>();

        public string ToJson(JsonWriter w)
        {
            w.Clear().BeginObject().Key("v").Value(Protocol.Version).Key("type").Value("tracking").Key("t_ns").Value(TNs);
            w.Key("head").Values(Head);
            w.Key("controllers").BeginObject().Key("left").Values(Left).Key("right").Values(Right).EndObject();
            w.Key("buttons").BeginObject()
                .Key("A").Value(A).Key("B").Value(B).Key("X").Value(X).Key("Y").Value(Y)
                .Key("left_trigger").Value(LeftTrigger).Key("right_trigger").Value(RightTrigger)
                .Key("left_grip").Value(LeftGrip).Key("right_grip").Value(RightGrip)
                .Key("left_menu").Value(LeftMenu).Key("right_menu").Value(RightMenu)
                .EndObject();
            w.Key("axes").BeginObject()
                .Key("left").BeginArray().Value(LX).Value(LY).EndArray()
                .Key("right").BeginArray().Value(RX).Value(RY).EndArray()
                .EndObject();
            w.Key("body");
            if (Body == null) w.Null();
            else
            {
                w.BeginArray();
                for (int j = 0; j < BodyJoints; j++)
                {
                    w.BeginArray();
                    for (int k = 0; k < 7; k++) w.Value(Body[7 * j + k]);
                    w.EndArray();
                }
                w.EndArray();
            }
            w.Key("body_ts");
            if (Body == null || BodyTs == null) w.Null();
            else
            {
                w.BeginArray();
                foreach (long t in BodyTs) w.Value(t);
                w.EndArray();
            }
            w.Key("trackers").BeginArray();
            for (int i = 0; i < TrackerSn.Count; i++)
                w.BeginObject().Key("sn").Value(TrackerSn[i]).Key("pose").Values(TrackerPose[i]).EndObject();
            w.EndArray();
            return w.EndObject().ToString();
        }
    }

    public struct SimEvent
    {
        public double T;
        public string Kind, Detail;
    }

    /// <summary>The server's "state" message.</summary>
    public sealed class SimState
    {
        public double T, Progress, CorrectMs, RetargetMs, PhysicsMs;
        public double HarnessPct = double.NaN, DataAge = double.NaN;
        public string Status = "", Message = "", Profile = "", Mode = "", Scene = "", GraspLeft, GraspRight;
        public double[] Qpos;
        public V3[] ObjPos = new V3[0];
        public Q4[] ObjQuat = new Q4[0];
        public Pose Rig = Pose.Identity, HeadCam = Pose.Identity;
        public bool Recording;
        public int Episodes;
        public long Frames;
        public readonly List<SimEvent> Events = new List<SimEvent>();

        public static SimState FromJson(Dictionary<string, object> d)
        {
            var s = new SimState
            {
                T = Json.Num(Json.Get(d, "t")),
                Status = Json.Str(Json.Get(d, "status"), ""),
                Message = Json.Str(Json.Get(d, "message"), ""),
                Progress = Json.Num(Json.Get(d, "progress")),
                Profile = Json.Str(Json.Get(d, "profile"), ""),
                Mode = Json.Str(Json.Get(d, "mode"), ""),
                Scene = Json.Str(Json.Get(d, "scene"), ""),
                Recording = Json.Bool(Json.Get(d, "recording")),
                Episodes = (int)Json.Num(Json.Get(d, "episodes")),
                Frames = (long)Json.Num(Json.Get(d, "frames")),
                HarnessPct = Json.Num(Json.Get(d, "harness_pct_mg"), double.NaN),
                DataAge = Json.Num(Json.Get(d, "data_age_s"), double.NaN),
            };
            s.Qpos = Json.Nums(Json.Get(Json.Obj(Json.Get(d, "robot")), "qpos"));
            var objs = Json.Obj(Json.Get(d, "objects"));
            var pos = Json.Arr(Json.Get(objs, "pos"));
            var quat = Json.Arr(Json.Get(objs, "quat"));
            if (pos != null && quat != null && pos.Count == quat.Count)
            {
                s.ObjPos = new V3[pos.Count];
                s.ObjQuat = new Q4[pos.Count];
                for (int i = 0; i < pos.Count; i++)
                {
                    s.ObjPos[i] = V3.From(Json.Nums(pos[i]));
                    s.ObjQuat[i] = Q4.From(Json.Nums(quat[i]));
                }
            }
            var grasp = Json.Obj(Json.Get(d, "grasp"));
            s.GraspLeft = Json.Str(Json.Get(grasp, "left"));
            s.GraspRight = Json.Str(Json.Get(grasp, "right"));
            s.Rig = PoseOf(Json.Get(d, "rig"));
            s.HeadCam = PoseOf(Json.Get(d, "head_cam"));
            var tm = Json.Obj(Json.Get(d, "timing_ms"));
            s.CorrectMs = Json.Num(Json.Get(tm, "correct_ms"));
            s.RetargetMs = Json.Num(Json.Get(tm, "retarget_ms"));
            s.PhysicsMs = Json.Num(Json.Get(tm, "physics_ms"));
            foreach (var e in Json.Arr(Json.Get(d, "events")) ?? new List<object>())
            {
                var ed = Json.Obj(e);
                s.Events.Add(new SimEvent { T = Json.Num(Json.Get(ed, "t")), Kind = Json.Str(Json.Get(ed, "kind"), ""), Detail = Json.Str(Json.Get(ed, "detail"), "") });
            }
            return s;
        }

        static Pose PoseOf(object o)
        {
            var d = Json.Obj(o);
            return d == null ? Pose.Identity : new Pose(V3.From(Json.Nums(Json.Get(d, "pos"))), Q4.From(Json.Nums(Json.Get(d, "quat"))).Normalized());
        }
    }

    /// <summary>The server's "scene" message.</summary>
    public sealed class SceneMessage
    {
        public string Name, Folder, RobotPath, Assets;   // Folder: scene files at <assets><Folder>/ (null: none)
        public int AssetPort;
        public double RateHz;
        public SceneSpec Spec;
        public readonly List<string> Dynamic = new List<string>();
        public readonly List<string> JointNames = new List<string>();

        public static SceneMessage FromJson(Dictionary<string, object> d)
        {
            var m = new SceneMessage
            {
                Name = Json.Str(Json.Get(d, "name"), ""),
                Folder = Json.Str(Json.Get(d, "folder")),
                RobotPath = Json.Str(Json.Get(d, "robot"), "_robot/robot.json"),
                Assets = Json.Str(Json.Get(d, "assets")),
                AssetPort = (int)Json.Num(Json.Get(d, "asset_port"), 9871),
                RateHz = Json.Num(Json.Get(d, "rate_hz"), 50),
                Spec = SceneSpec.FromJson(Json.Obj(Json.Get(d, "spec"))),
            };
            foreach (var n in Json.Arr(Json.Get(d, "dynamic")) ?? new List<object>()) m.Dynamic.Add(Json.Str(n, ""));
            foreach (var n in Json.Arr(Json.Get(d, "joint_names")) ?? new List<object>()) m.JointNames.Add(Json.Str(n, ""));
            return m;
        }

        /// <summary>Asset base URL on the host the app reached the server at (the advertised one may be another NIC).</summary>
        public string AssetBase(string serverHost)
        {
            return string.IsNullOrEmpty(serverHost) ? Assets : "http://" + serverHost + ":" + AssetPort + "/";
        }
    }

    /// <summary>
    /// Recent states with an estimate of the server clock, so the robot and objects can be drawn every
    /// display frame (72-90 Hz) by interpolating the 50 Hz states `Delay` seconds in the past.
    /// </summary>
    public sealed class StateBuffer
    {
        public double Delay = 0.04;
        readonly List<SimState> states = new List<SimState>();
        readonly List<double> offsets = new List<double>();   // local receive time - server time
        const int Keep = 8, OffsetWindow = 100;

        public SimState Latest { get { return states.Count > 0 ? states[states.Count - 1] : null; } }

        public void Clear() { states.Clear(); offsets.Clear(); }

        public void Add(SimState s, double localTime)
        {
            if (states.Count > 0 && s.T <= Latest.T) return;  // late / duplicate datagram
            states.Add(s);
            if (states.Count > Keep) states.RemoveAt(0);
            offsets.Add(localTime - s.T);
            if (offsets.Count > OffsetWindow) offsets.RemoveAt(0);
        }

        /// <summary>Smallest observed delay: the receive time of the least delayed datagram.</summary>
        double Offset()
        {
            double m = double.MaxValue;
            foreach (double o in offsets) m = Math.Min(m, o);
            return m;
        }

        /// <summary>Interpolated robot qpos and object poses at localTime - Delay. False if no state yet.</summary>
        public bool Sample(double localTime, double[] qpos, V3[] objPos, Q4[] objQuat)
        {
            if (states.Count == 0) return false;
            double t = localTime - Delay - Offset();
            SimState a = states[0], b = states[0];
            double u = 0;
            if (t >= Latest.T) a = b = Latest;
            else if (t > states[0].T)
            {
                for (int i = 1; i < states.Count; i++)
                {
                    if (states[i].T >= t)
                    {
                        a = states[i - 1];
                        b = states[i];
                        u = (t - a.T) / Math.Max(1e-6, b.T - a.T);
                        break;
                    }
                }
            }
            if (a.Qpos == null || b.Qpos == null || a.Qpos.Length != b.Qpos.Length) a = b;
            if (b.Qpos != null && qpos != null)
            {
                int n = Math.Min(qpos.Length, b.Qpos.Length);
                for (int i = 0; i < n; i++) qpos[i] = a.Qpos[i] + (b.Qpos[i] - a.Qpos[i]) * u;
                if (n >= 7)
                {
                    Q4 q = Q4.Slerp(Q4.From(a.Qpos, 3).Normalized(), Q4.From(b.Qpos, 3).Normalized(), u);
                    qpos[3] = q.w; qpos[4] = q.x; qpos[5] = q.y; qpos[6] = q.z;
                }
            }
            bool sameObjects = a.ObjPos.Length == b.ObjPos.Length;
            int no = Math.Min(objPos == null ? 0 : objPos.Length, b.ObjPos.Length);
            for (int i = 0; i < no; i++)
            {
                objPos[i] = sameObjects ? V3.Lerp(a.ObjPos[i], b.ObjPos[i], u) : b.ObjPos[i];
                objQuat[i] = sameObjects ? Q4.Slerp(a.ObjQuat[i].Normalized(), b.ObjQuat[i].Normalized(), u) : b.ObjQuat[i].Normalized();
            }
            return true;
        }
    }
}
