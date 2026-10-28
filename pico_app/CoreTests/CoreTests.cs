// Tests of the app's Unity-independent core (Assets/IgrisPico/Core) against fixtures from the Python
// server code (MuJoCo FK, scene rotations, frame conversions). Run by tests/test_app.py:
//   csc -out:CoreTests.exe CoreTests.cs ../Assets/IgrisPico/Core/*.cs
//   CoreTests.exe <fixtures.json> <output dir>
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using IgrisPico.Core;

static class CoreTests
{
    static int failures;
    static string fixtureDir;

    static int Main(string[] args)
    {
        var fx = Json.Obj(Json.Parse(File.ReadAllText(args[0], Encoding.UTF8)));
        fixtureDir = Path.GetDirectoryName(Path.GetFullPath(args[0]));
        string outDir = args[1];
        Run("json", () => TestJson());
        Run("conversions", () => TestConversions(fx));
        Run("robot_fk", () => TestRobotFk(fx));
        Run("robot_meshes", () => TestRobotMeshes(fx));
        Run("mesh_winding", () => TestMeshWinding());
        Run("stl", () => TestStl());
        Run("scenes", () => TestScenes(fx));
        Run("state", () => TestState(fx));
        Run("state_buffer", () => TestStateBuffer());
        Run("scene_message", () => TestSceneMessage(fx));
        Run("tracking", () => TestTracking(fx, outDir));
        Run("sim_link", () => TestSimLink());
        Console.WriteLine(failures == 0 ? "ALL PASSED" : failures + " FAILED");
        return failures == 0 ? 0 : 1;
    }

    static void Run(string name, Action test)
    {
        try { test(); Console.WriteLine("PASS " + name); }
        catch (Exception e) { failures++; Console.WriteLine("FAIL " + name + ": " + e); }
    }

    static void Check(bool ok, string what) { if (!ok) throw new Exception(what); }

    static void Near(double a, double b, double tol, string what)
    {
        if (!(Math.Abs(a - b) <= tol)) throw new Exception(string.Format("{0}: {1} != {2} (tol {3})", what, a, b, tol));
    }

    static void NearV(V3 a, double[] b, double tol, string what)
    {
        Near(a.x, b[0], tol, what + ".x"); Near(a.y, b[1], tol, what + ".y"); Near(a.z, b[2], tol, what + ".z");
    }

    static void SameRot(Q4 a, double[] wxyz, double tol, string what)
    {
        Near(Math.Abs(Q4.Dot(a.Normalized(), Q4.From(wxyz).Normalized())), 1.0, tol, what);
    }

    static List<object> L(Dictionary<string, object> d, string k) { return Json.Arr(Json.Get(d, k)); }

    // ---------------------------------------------------------------------------------------
    static void TestJson()
    {
        string src = "{\"a\": [1, -2.5e-3, 3E2, true, false, null], \"s\": \"q\\\"\\\\\\n\\u00e9\", \"o\": {}, \"e\": []}";
        var d = Json.Obj(Json.Parse(src));
        var a = L(d, "a");
        Near((double)a[0], 1, 0, "int"); Near((double)a[1], -0.0025, 1e-15, "exp"); Near((double)a[2], 300, 0, "Exp");
        Check((bool)a[3] && !(bool)a[4] && a[5] == null, "literals");
        Check(Json.Str(Json.Get(d, "s")) == "q\"\\\né", "escapes");
        Check(Json.Obj(Json.Get(d, "o")).Count == 0 && L(d, "e").Count == 0, "empty containers");
        var w = new JsonWriter().BeginObject().Key("x").Value(0.1).Key("s").Value("a\"b\n").Key("n").Null()
            .Key("l").BeginArray().Value(1).Value(true).BeginObject().EndObject().EndArray().Key("big").Value(1761100000000000000L).EndObject();
        var back = Json.Obj(Json.Parse(w.ToString()));
        Near(Json.Num(Json.Get(back, "x")), 0.1, 0, "round trip double");
        Check(Json.Str(Json.Get(back, "s")) == "a\"b\n", "round trip string");
        Check(Json.Get(back, "n") == null && L(back, "l").Count == 3, "round trip structure");
        Near(Json.Num(Json.Get(back, "big")), 1761100000000000000.0, 1e3, "long");
        bool threw = false;
        try { Json.Parse("{\"a\": 1,}"); } catch (FormatException) { threw = true; }
        Check(threw, "rejects trailing comma");
    }

    static void TestConversions(Dictionary<string, object> fx)
    {
        foreach (var o in L(fx, "unity_from_mj"))
        {
            var c = Json.Obj(o);
            V3 up = Conv.UnityFromMj(V3.From(Json.Nums(Json.Get(c, "p"))));
            NearV(up, Json.Nums(Json.Get(c, "up")), 1e-12, "unity pos");
            UQuat uq = Conv.UnityFromMj(Q4.From(Json.Nums(Json.Get(c, "q"))));
            double[] e = Json.Nums(Json.Get(c, "uq"));
            Near(uq.x, e[0], 1e-12, "uq.x"); Near(uq.y, e[1], 1e-12, "uq.y"); Near(uq.z, e[2], 1e-12, "uq.z"); Near(uq.w, e[3], 1e-12, "uq.w");
        }
        foreach (var o in L(fx, "native_from_unity"))
        {
            var c = Json.Obj(o);
            double[] p = Json.Nums(Json.Get(c, "p")), q = Json.Nums(Json.Get(c, "q")), n = Json.Nums(Json.Get(c, "n"));
            double[] got = Conv.NativeFromUnity(p[0], p[1], p[2], q[0], q[1], q[2], q[3]);
            for (int k = 0; k < 7; k++) Near(got[k], n[k], 1e-12, "native[" + k + "]");
        }
        foreach (var o in L(fx, "euler"))
        {
            var c = Json.Obj(o);
            double[] deg = Json.Nums(Json.Get(c, "deg"));
            SameRot(Conv.EulerXyzDeg(deg[0], deg[1], deg[2]), Json.Nums(Json.Get(c, "quat")), 1e-12, "euler");
        }
    }

    static RobotModel LoadRobot(Dictionary<string, object> fx)
    {
        string dir = Path.Combine(fixtureDir, Json.Str(Json.Get(fx, "robot_dir")));
        return RobotModel.Parse(File.ReadAllText(Path.Combine(dir, "robot.json")));
    }

    static void TestRobotFk(Dictionary<string, object> fx)
    {
        RobotModel r = LoadRobot(fx);
        Check(r.Nq == 38 && r.JointNames.Count == 31, "nq / joints");
        var world = new Pose[r.Bodies.Count];
        int cases = 0;
        foreach (var o in L(fx, "fk"))
        {
            var c = Json.Obj(o);
            r.ForwardKinematics(Json.Nums(Json.Get(c, "qpos")), world);
            var bp = L(c, "body_pos"); var bq = L(c, "body_quat");
            for (int k = 0; k < r.Bodies.Count; k++)
            {
                NearV(world[k].p, Json.Nums(bp[k]), 1e-9, "body " + r.Bodies[k].Name + " pos");
                SameRot(world[k].q, Json.Nums(bq[k]), 1e-9, "body " + r.Bodies[k].Name + " quat");
            }
            var gp = L(c, "geom_pos"); var gq = L(c, "geom_quat");
            for (int g = 0; g < r.Geoms.Count; g++)
            {
                Pose x = world[r.Geoms[g].Body] * r.Geoms[g].Local;
                NearV(x.p, Json.Nums(gp[g]), 2e-6, "geom " + r.Geoms[g].Name + " pos");   // robot.json rounds to 1e-6
                SameRot(x.q, Json.Nums(gq[g]), 1e-9, "geom " + r.Geoms[g].Name + " quat");
            }
            cases++;
        }
        Check(cases >= 2, "fk cases");
        Check(r.Descends(r.BodyIndex("Link_Neck_Pitch"), r.BodyIndex("base_link")), "descends");
    }

    static void TestRobotMeshes(Dictionary<string, object> fx)
    {
        RobotModel r = LoadRobot(fx);
        string dir = Path.Combine(fixtureDir, Json.Str(Json.Get(fx, "robot_dir")));
        var counts = Json.Obj(Json.Get(fx, "mesh_counts"));
        foreach (var g in r.Geoms.Where(g => g.Type == "mesh"))
        {
            MeshData m = MeshIO.Load(g.Mesh, File.ReadAllBytes(Path.Combine(dir, g.Mesh)));
            double[] n = Json.Nums(Json.Get(counts, g.Mesh));
            Check(m.Pos.Count == (int)n[0] && m.TriangleCount == (int)n[1], g.Mesh + " counts");
            UnityMeshData u = MeshIO.ToUnity(m, new V3(1, 1, 1));
            Check(u.Indices.Length == 3 * m.TriangleCount && u.VertexCount >= m.Pos.Count, g.Mesh + " unity data");
        }
    }

    // A closed cube: after conversion every face must face outward (Unity winding) and the 40-degree crease must
    // keep the 6 faces flat (24 vertices).
    static string CubeObj()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < 8; i++)
            sb.AppendFormat("v {0} {1} {2}\n", (i & 1) != 0 ? 1.5 : 0.5, (i & 2) != 0 ? 0.5 : -0.5, (i & 4) != 0 ? 0.25 : -0.25);
        // counter-clockwise seen from outside (right-handed, as MuJoCo / OBJ)
        sb.Append("f 1 3 4 2\nf 5 6 8 7\nf 1 2 6 5\nf 3 7 8 4\nf 1 5 7 3\nf 2 4 8 6\n");
        return sb.ToString();
    }

    static void TestMeshWinding()
    {
        MeshData m = MeshIO.ParseObj(CubeObj());
        Check(m.Pos.Count == 8 && m.TriangleCount == 12, "cube counts");
        var center = V3.Zero;
        foreach (V3 p in m.Pos) center = center + p * (1.0 / 8);
        // outward in the source frame
        for (int f = 0; f < 12; f++)
        {
            V3 a = m.Pos[m.Tri[3 * f]], b = m.Pos[m.Tri[3 * f + 1]], c = m.Pos[m.Tri[3 * f + 2]];
            Check(V3.Dot(V3.Cross(b - a, c - a), (a + b + c) * (1.0 / 3) - center) > 0, "source cube is outward");
        }
        var scale = new V3(2, 1, 1);
        UnityMeshData u = MeshIO.ToUnity(m, scale);
        Check(u.VertexCount == 24, "crease keeps faces flat: " + u.VertexCount);
        V3 cu = Conv.UnityFromMj(new V3(center.x * 2, center.y, center.z));
        Func<int, V3> P = i => new V3(u.Vertices[3 * i], u.Vertices[3 * i + 1], u.Vertices[3 * i + 2]);
        Func<int, V3> N = i => new V3(u.Normals[3 * i], u.Normals[3 * i + 1], u.Normals[3 * i + 2]);
        for (int f = 0; f < 12; f++)
        {
            V3 a = P(u.Indices[3 * f]), b = P(u.Indices[3 * f + 1]), c = P(u.Indices[3 * f + 2]);
            V3 outward = (a + b + c) * (1.0 / 3) - cu;
            // Unity front faces: the numeric cross product (b - a) x (c - a) points to the viewer (outside)
            Check(V3.Dot(V3.Cross(b - a, c - a), outward) > 0, "unity face " + f + " is outward");
            for (int k = 0; k < 3; k++) Check(V3.Dot(N(u.Indices[3 * f + k]), outward) > 0, "normal outward");
        }
        // vertex 0 = (0.5, -0.5, -0.25) scaled (x2) -> Unity (0.5, -0.25, 1.0)
        bool found = false;
        for (int i = 0; i < u.VertexCount; i++)
            if ((P(i) - new V3(0.5, -0.25, 1.0)).Norm < 1e-6) found = true;
        Check(found, "converted vertex position");
    }

    static void TestStl()
    {
        MeshData cube = MeshIO.ParseObj(CubeObj());
        var bin = new MemoryStream();
        var w = new BinaryWriter(bin);
        w.Write(new byte[80]);
        w.Write((uint)cube.TriangleCount);
        var ascii = new StringBuilder("solid cube\n");
        for (int f = 0; f < cube.TriangleCount; f++)
        {
            w.Write(0f); w.Write(0f); w.Write(0f);
            ascii.Append("facet normal 0 0 0\nouter loop\n");
            for (int k = 0; k < 3; k++)
            {
                V3 p = cube.Pos[cube.Tri[3 * f + k]];
                w.Write((float)p.x); w.Write((float)p.y); w.Write((float)p.z);
                ascii.AppendFormat(System.Globalization.CultureInfo.InvariantCulture, "vertex {0} {1} {2}\n", p.x, p.y, p.z);
            }
            w.Write((ushort)0);
            ascii.Append("endloop\nendfacet\n");
        }
        ascii.Append("endsolid cube\n");
        foreach (var m in new[] { MeshIO.ParseStl(bin.ToArray()), MeshIO.ParseStl(Encoding.ASCII.GetBytes(ascii.ToString())) })
        {
            Check(m.Pos.Count == 8 && m.TriangleCount == 12, "stl welded cube");
            for (int k = 0; k < m.Tri.Count; k++)
                Check((m.Pos[m.Tri[k]] - cube.Pos[cube.Tri[k]]).Norm < 1e-6, "stl corner " + k);
        }
    }

    static void TestScenes(Dictionary<string, object> fx)
    {
        int n = 0;
        foreach (var o in L(fx, "scenes"))
        {
            var c = Json.Obj(o);
            SceneSpec s = SceneSpec.FromJson(Json.Obj(Json.Get(c, "spec")));
            var expected = L(c, "objects");
            Check(s.Objects.Count == expected.Count, s.Name + " object count");
            for (int i = 0; i < expected.Count; i++)
            {
                var e = Json.Obj(expected[i]);
                SceneObject obj = s.Objects[i];
                Check(obj.Name == Json.Str(Json.Get(e, "name")), "name");
                NearV(obj.Pose.p, Json.Nums(Json.Get(e, "pos")), 1e-12, obj.Name + " pos");
                SameRot(obj.Pose.q, Json.Nums(Json.Get(e, "quat")), 1e-12, obj.Name + " quat");
                Check(obj.Dynamic == Json.Bool(Json.Get(e, "dynamic")) && obj.Graspable == Json.Bool(Json.Get(e, "graspable")), obj.Name + " flags");
                var parts = L(e, "parts");
                Check(obj.Parts.Count == parts.Count, obj.Name + " parts");
                for (int k = 0; k < parts.Count; k++)
                {
                    var pe = Json.Obj(parts[k]);
                    ScenePart p = obj.Parts[k];
                    Check(p.Shape == Json.Str(Json.Get(pe, "shape")), "shape");
                    NearV(p.Local.p, Json.Nums(Json.Get(pe, "pos")), 1e-12, obj.Name + " part pos");
                    SameRot(p.Local.q, Json.Nums(Json.Get(pe, "quat")), 1e-12, obj.Name + " part quat");
                    double[] rgba = Json.Nums(Json.Get(pe, "rgba"));
                    for (int j = 0; j < 4; j++) Near(p.Rgba[j], rgba[j], 1e-12, "rgba");
                    double[] us = Json.Nums(Json.Get(pe, "unity_scale"));
                    if (us != null) NearV(SceneSpec.UnityPrimitiveScale(p.Shape, p.Size), us, 1e-12, obj.Name + " unity scale");
                }
            }
            n++;
        }
        Check(n >= 3, "scene count");
    }

    static void TestState(Dictionary<string, object> fx)
    {
        var d = Json.Obj(Json.Get(fx, "state"));
        SimState s = SimState.FromJson(d);
        double[] q = Json.Nums(Json.Get(Json.Obj(Json.Get(d, "robot")), "qpos"));
        Check(s.Qpos.Length == q.Length && s.Status == Json.Str(Json.Get(d, "status")), "qpos / status");
        Check(s.ObjPos.Length == L(Json.Obj(Json.Get(d, "objects")), "pos").Count, "objects");
        NearV(s.Rig.p, Json.Nums(Json.Get(Json.Obj(Json.Get(d, "rig")), "pos")), 1e-12, "rig");
        Check(double.IsNaN(s.HarnessPct) == (Json.Get(d, "harness_pct_mg") == null), "harness null");
    }

    static SimState FakeState(double t, double x)
    {
        var s = new SimState { T = t, Qpos = new double[38], ObjPos = new[] { new V3(x, 0, 0) }, ObjQuat = new[] { Q4.Identity } };
        s.Qpos[0] = x; s.Qpos[3] = 1; s.Qpos[7] = 2 * x;
        return s;
    }

    static void TestStateBuffer()
    {
        var b = new StateBuffer { Delay = 0.04 };
        var q = new double[38];
        var op = new V3[1];
        var oq = new Q4[1];
        Check(!b.Sample(0, q, op, oq), "empty buffer");
        // server clock 1000 s ahead of the local clock, 10 ms network delay
        for (int i = 0; i <= 10; i++) b.Add(FakeState(1000 + 0.02 * i, 0.1 * i), 0.02 * i + 0.010);
        b.Add(FakeState(1000.1, 9), 0.5);   // late duplicate is ignored
        Check(b.Sample(0.2 + 0.010 + 0.04 - 0.01, q, op, oq), "sample");   // server time 1000.19
        Near(q[0], 0.95, 1e-9, "interpolated root x");
        Near(q[7], 1.9, 1e-9, "interpolated joint");
        Near(op[0].x, 0.95, 1e-9, "interpolated object");
        b.Sample(5.0, q, op, oq);
        Near(q[0], 1.0, 1e-12, "holds the latest state");
    }

    static void TestSceneMessage(Dictionary<string, object> fx)
    {
        SceneMessage m = SceneMessage.FromJson(Json.Obj(Json.Get(fx, "scene_message")));
        Check(m.Name == "tabletop" && m.Spec.Objects.Count > 0 && m.JointNames.Count == 31, "scene message");
        Check(m.RobotPath == "_robot/robot.json" && m.Folder == "tabletop", "robot path / scene folder");
        Check(m.AssetBase("10.0.0.5") == "http://10.0.0.5:" + m.AssetPort + "/", "asset base");
        Check(m.Dynamic.Count > 0 && m.Dynamic.TrueForAll(n => m.Spec.Objects.Any(o => o.Name == n && o.Dynamic)), "dynamic names");
    }

    static void TestTracking(Dictionary<string, object> fx, string outDir)
    {
        var c = Json.Obj(Json.Get(fx, "tracking_unity"));
        var f = new TrackingFrame { TNs = 1761100000123456789L, A = true, Y = true, RightGrip = 0.8, LeftTrigger = 0.25, LX = -0.5, RY = 0.75 };
        Action<string, double[]> pose = (k, dst) =>
        {
            double[] u = Json.Nums(Json.Get(c, k));
            Conv.NativeFromUnity(u[0], u[1], u[2], u[3], u[4], u[5], u[6], dst);
        };
        pose("head", f.Head);
        pose("left", f.Left);
        pose("right", f.Right);
        f.Body = new double[24 * 7];
        f.BodyTs = new long[24];
        for (int j = 0; j < 24; j++)
        {
            Conv.NativeFromUnity(0.01 * j, 1.0 - 0.03 * j, 0.2, 0, 0, 0, 1, f.Body, 7 * j);
            f.BodyTs[j] = 123456 + j;
        }
        var tp = new double[7];
        pose("tracker", tp);
        f.TrackerSn.Add("PC2310MLL4210362G");
        f.TrackerPose.Add(tp);
        File.WriteAllText(Path.Combine(outDir, "tracking.json"), f.ToJson(new JsonWriter()));
        File.WriteAllText(Path.Combine(outDir, "cmd.json"), Protocol.Cmd("record", "on", false, "scene", "room"));
    }

    // SimLink (Runtime/SimLink.cs has no Unity dependency) against a loopback "server"
    static void TestSimLink()
    {
        var server = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0));
        server.Client.ReceiveTimeout = 2000;
        int port = ((IPEndPoint)server.Client.LocalEndPoint).Port;
        using (var link = new IgrisPico.SimLink(port))
        {
            Check(!link.Connected && link.SetServer("127.0.0.1") && !link.SetServer("not-an-ip") && link.Server == null, "set server");
            link.SetServer("127.0.0.1");
            link.Send(Protocol.Cmd("hello"));
            var from = new IPEndPoint(IPAddress.Any, 0);
            Check(Encoding.UTF8.GetString(server.Receive(ref from)).Contains("\"hello\""), "server got hello");
            byte[] reply = Encoding.UTF8.GetBytes("{\"v\":1,\"type\":\"ack\",\"cmd\":\"hello\"}");
            server.Send(reply, reply.Length, from);
            IgrisPico.SimLink.Packet p;
            var t0 = DateTime.UtcNow;
            while (!link.TryReceive(out p))
            {
                if ((DateTime.UtcNow - t0).TotalSeconds > 2) throw new Exception("no reply received");
                Thread.Sleep(5);
            }
            Check(p.Text.Contains("ack") && link.Accept(p) && link.Connected, "reply accepted");
            var other = new IgrisPico.SimLink.Packet { Text = "{}", From = new IPEndPoint(IPAddress.Parse("10.1.2.3"), port), Time = IgrisPico.AppClock.Now };
            Check(!link.Accept(other), "a second server is ignored while connected");
        }
        server.Close();
        long a = IgrisPico.AppClock.Ns;
        Thread.Sleep(2);
        Check(IgrisPico.AppClock.Ns > a && a > 1600000000000000000L, "clock in ns since the epoch");
    }
}
