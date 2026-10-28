// IGRIS-C teleoperation client for PICO 4 Ultra (docs/sim_protocol.md).
// Every frame: headset / controller / motion-tracker poses -> "tracking" to the PC simulation server;
// "state" from the server (50 Hz) -> robot, objects and rig, interpolated to the display rate.
// The scene and the robot are built from the server's scene message and asset server, so the app needs
// no per-scene content.
using System.Collections;
using System.Collections.Generic;
using IgrisPico.Core;
using UnityEngine;
using UnityEngine.XR;

namespace IgrisPico
{
    public sealed class AppMain : MonoBehaviour
    {
        [Tooltip("XR rig root (tracking space origin). Built at runtime when empty.")]
        public Transform rig;
        [Tooltip("Camera driven by the headset (child of the rig's camera offset).")]
        public Camera xrCamera;

        Settings settings;
        SimLink link;
        readonly HeadsetInput input = new HeadsetInput();
        readonly PicoTracking pico = new PicoTracking();
        readonly StateBuffer states = new StateBuffer();
        readonly AssetCache cache = new AssetCache();
        readonly TrackingFrame frame = new TrackingFrame();
        readonly JsonWriter writer = new JsonWriter();
        readonly double[] body = new double[TrackingFrame.BodyJoints * 7];
        readonly long[] bodyTs = new long[TrackingFrame.BodyJoints];
        Transform cameraOffset, leftHand, rightHand;
        RobotView robot;
        SceneView scene;
        RigView rigView;
        Hud hud;

        SceneMessage sceneMsg;
        string robotHost, buildingScene, sceneMenuName;
        bool loadingRobot;
        readonly List<string> sceneNames = new List<string>();
        double[] qpos = new double[38];
        V3[] objPos = new V3[0];
        Q4[] objQuat = new Q4[0];
        float originRetryUntil;
        bool originSet;
        static readonly string Hello = Protocol.Cmd("hello");

        void Awake()
        {
            BuildRig();
            var light = new GameObject("Sun").AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.0f;
            light.shadows = LightShadows.None;
            light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.48f, 0.5f, 0.53f);
            var world = new GameObject("World").transform;
            scene = new SceneView(world);
            robot = new RobotView(world);
            rigView = new RigView(rig, xrCamera.transform);
            hud = new Hud(xrCamera.transform, leftHand);
        }

        void BuildRig()
        {
            if (rig == null) rig = new GameObject("XR Rig").transform;
            if (xrCamera == null)
            {
                cameraOffset = new GameObject("Camera Offset").transform;
                cameraOffset.SetParent(rig, false);
                var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
                camGo.transform.SetParent(cameraOffset, false);
                xrCamera = camGo.AddComponent<Camera>();
                camGo.AddComponent<AudioListener>();
                var tpd = camGo.AddComponent<UnityEngine.SpatialTracking.TrackedPoseDriver>();
                tpd.SetPoseSource(UnityEngine.SpatialTracking.TrackedPoseDriver.DeviceType.GenericXRDevice,
                                  UnityEngine.SpatialTracking.TrackedPoseDriver.TrackedPose.Center);
                tpd.trackingType = UnityEngine.SpatialTracking.TrackedPoseDriver.TrackingType.RotationAndPosition;
                tpd.updateType = UnityEngine.SpatialTracking.TrackedPoseDriver.UpdateType.UpdateAndBeforeRender;
            }
            cameraOffset = xrCamera.transform.parent;
            xrCamera.nearClipPlane = 0.03f;
            xrCamera.farClipPlane = 200f;
            xrCamera.clearFlags = CameraClearFlags.SolidColor;
            xrCamera.backgroundColor = new Color(0.62f, 0.72f, 0.82f);
            leftHand = Controller("Left Controller", new Color(0.35f, 0.55f, 0.95f));
            rightHand = Controller("Right Controller", new Color(0.95f, 0.45f, 0.35f));
        }

        Transform Controller(string name, Color c)
        {
            var t = new GameObject(name).transform;
            t.SetParent(cameraOffset, false);
            GameObject box = Visuals.Primitive(PrimitiveType.Cube, t, Visuals.Lit(c), "Model");
            box.transform.localScale = new Vector3(0.035f, 0.03f, 0.11f);
            box.transform.localPosition = new Vector3(0f, -0.01f, -0.03f);
            return t;
        }

        void Start()
        {
            settings = Settings.Load();
            link = new SimLink(settings.Port);
            link.SetServer(settings.Server);
            pico.ObjectCount = settings.ObjectTrackers;
            pico.SetMode(settings.Trackers);
            rigView.Mode = settings.View;
            originRetryUntil = Time.unscaledTime + 10f;
            BuildMenu();
            Debug.Log("[IGRIS] settings from " + settings.Source + ", server " + (settings.Server == "" ? "auto" : settings.Server));
        }

        void OnDestroy()
        {
            if (link != null) link.Dispose();
            pico.SetMode(TrackerMode.None);
        }

        void SetFloorOrigin()
        {
            var subsystems = new List<XRInputSubsystem>();
            SubsystemManager.GetInstances(subsystems);
            foreach (XRInputSubsystem s in subsystems)
                if (s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Floor)) originSet = true;
        }

        void Update()
        {
            float dt = Time.unscaledDeltaTime;
            if (!originSet && Time.unscaledTime < originRetryUntil) SetFloorOrigin();
            input.Poll();
            if (input.Simulated)  // Editor without headset: drive the camera from the simulated head
            {
                xrCamera.transform.localPosition = input.HeadPos;
                xrCamera.transform.localRotation = input.HeadRot;
            }
            Place(leftHand, input.Left);
            Place(rightHand, input.Right);
            bool menuHasStick = hud.HandleInput(input.Left);

            SendTracking(menuHasStick);
            link.Discover(Hello);
            Receive();

            double now = AppClock.Now;
            if (states.Latest != null)
            {
                SimState s = states.Latest;
                if (s.Qpos != null && qpos.Length != s.Qpos.Length) qpos = new double[s.Qpos.Length];
                if (objPos.Length != s.ObjPos.Length)
                {
                    objPos = new V3[s.ObjPos.Length];
                    objQuat = new Q4[s.ObjPos.Length];
                }
                if (states.Sample(now, qpos, objPos, objQuat))
                {
                    robot.Apply(qpos);
                    if (scene.Name == s.Scene) scene.Apply(objPos, objQuat, objPos.Length);
                }
                scene.Highlight(s.GraspLeft, s.GraspRight);
                rigView.Apply(s, dt);
            }
            robot.SetStyle(settings.GhostRobot, rigView.Mode != ViewMode.Embodied);
            hud.Draw(StatusText(), NoticeText(), states.Latest != null && states.Latest.Recording, dt);
        }

        static void Place(Transform t, ControllerState c)
        {
            t.gameObject.SetActive(c.Valid);
            if (!c.Valid) return;
            t.localPosition = c.Pos;
            t.localRotation = c.Rot;
        }

        // -- tracking out ----------------------------------------------------------------------
        void SendTracking(bool menuHasStick)
        {
            if (link.Server == null) return;
            TrackingFrame f = frame;
            f.TNs = AppClock.Ns;
            Native(input.HeadValid, input.HeadPos, input.HeadRot, f.Head);
            Native(input.Left.Valid, input.Left.Pos, input.Left.Rot, f.Left);
            Native(input.Right.Valid, input.Right.Pos, input.Right.Rot, f.Right);
            f.A = input.Right.Primary;
            f.B = input.Right.Secondary;
            f.X = input.Left.Primary;
            f.Y = input.Left.Secondary;
            f.LeftMenu = input.Left.Menu;
            f.RightMenu = input.Right.Menu;
            f.LeftTrigger = menuHasStick ? 0f : input.Left.Trigger;   // the left trigger runs menu items
            f.RightTrigger = input.Right.Trigger;
            f.LeftGrip = input.Left.Grip;
            f.RightGrip = input.Right.Grip;
            f.LX = menuHasStick ? 0f : input.Left.Stick.x;
            f.LY = menuHasStick ? 0f : input.Left.Stick.y;
            f.RX = input.Right.Stick.x;
            f.RY = input.Right.Stick.y;
            bool newBody = pico.ReadBody(body, bodyTs);
            f.Body = newBody ? body : null;
            f.BodyTs = newBody ? bodyTs : null;
            pico.ReadObjects(f.TrackerSn, f.TrackerPose);
            link.Send(f.ToJson(writer));
        }

        static void Native(bool valid, Vector3 p, Quaternion q, double[] dst)
        {
            if (!valid)
            {
                System.Array.Clear(dst, 0, 7);   // all zeros = not tracked
                return;
            }
            Conv.NativeFromUnity(p.x, p.y, p.z, q.x, q.y, q.z, q.w, dst);
        }

        // -- messages in ------------------------------------------------------------------------
        void Receive()
        {
            SimLink.Packet p;
            int n = 0;
            while (n++ < 200 && link.TryReceive(out p))
            {
                Dictionary<string, object> msg;
                try { msg = Json.Obj(Json.Parse(p.Text)); }
                catch (System.Exception) { continue; }   // not JSON
                if (msg == null || (int)Json.Num(Json.Get(msg, "v"), Protocol.Version) != Protocol.Version) continue;
                bool wasConnected = link.Connected;
                if (!link.Accept(p)) continue;   // another server on the LAN
                if (!wasConnected)
                {
                    link.Send(Protocol.Cmd("list_scenes"));
                    if (sceneMsg == null) link.Send(Hello);
                }
                try { Handle(msg, p.Time); }
                catch (System.Exception e)
                {
                    Debug.LogWarning("[IGRIS] bad " + Protocol.TypeOf(msg) + " message: " + e);
                    hud.Toast("Bad " + Protocol.TypeOf(msg) + " message: " + e.Message, 5f);
                }
            }
        }

        void Handle(Dictionary<string, object> msg, double time)
        {
            switch (Protocol.TypeOf(msg))
            {
                case "state":
                    states.Add(SimState.FromJson(msg), time);
                    break;
                case "scene":
                    OnScene(SceneMessage.FromJson(msg));
                    break;
                case "scenes":
                    sceneNames.Clear();
                    foreach (object o in Json.Arr(Json.Get(msg, "names")) ?? new List<object>()) sceneNames.Add(Json.Str(o, ""));
                    if (sceneMenuName == null) sceneMenuName = Json.Str(Json.Get(msg, "current"));
                    break;
                case "ack":
                    hud.Toast("OK: " + Json.Str(Json.Get(msg, "cmd"), ""));
                    break;
                case "error":
                    hud.Toast("Error: " + Json.Str(Json.Get(msg, "error"), ""), 5f);
                    break;
            }
        }

        void OnScene(SceneMessage msg)
        {
            sceneMsg = msg;
            sceneMenuName = msg.Name;
            string assets = msg.AssetBase(link.ServerHost);
            if (robotHost != link.ServerHost && !loadingRobot) StartCoroutine(LoadRobot(assets, msg.RobotPath));
            if (scene.Name != msg.Name && buildingScene != msg.Name)
            {
                states.Clear();
                rigView.Reset();
                StartCoroutine(BuildScene(msg, assets));
            }
        }

        IEnumerator LoadRobot(string assets, string path)
        {
            loadingRobot = true;
            string host = link.ServerHost;
            byte[] data = null;
            yield return cache.Get(assets + path, b => data = b);
            if (data != null)
            {
                RobotModel model = null;
                try { model = RobotModel.Parse(System.Text.Encoding.UTF8.GetString(data)); }
                catch (System.Exception e) { hud.Toast("robot.json: " + e.Message, 5f); }
                if (model != null)
                {
                    string dir = path.Contains("/") ? path.Substring(0, path.LastIndexOf('/') + 1) : "";
                    yield return robot.Build(model, assets + dir, cache);
                    robotHost = host;
                }
            }
            loadingRobot = false;
        }

        IEnumerator BuildScene(SceneMessage msg, string assets)
        {
            buildingScene = msg.Name;
            yield return scene.Build(msg, assets, cache);
            if (buildingScene == msg.Name) buildingScene = null;
            hud.Toast("Scene: " + msg.Name);
        }

        // -- HUD ----------------------------------------------------------------------------------
        string StatusText()
        {
            var sb = new System.Text.StringBuilder();
            sb.Append("IGRIS sim   ").Append(link.Connected ? "server " + link.ServerHost : "searching server...").Append('\n');
            SimState s = states.Latest;
            if (s != null)
            {
                sb.Append(s.Status == "calibrating" ? "CALIBRATING " + (int)(s.Progress * 100) + "%" : s.Status.ToUpperInvariant())
                  .Append("   profile ").Append(string.IsNullOrEmpty(s.Profile) ? "-" : s.Profile).Append('\n');
                sb.Append("scene ").Append(s.Scene).Append("   physics ").Append(s.Mode).Append('\n');
                sb.Append(s.Recording ? "REC  " : "rec off  ").Append("episodes ").Append(s.Episodes).Append('\n');
                sb.Append("grasp L ").Append(s.GraspLeft ?? "-").Append("  R ").Append(s.GraspRight ?? "-").Append('\n');
                sb.Append("server ik ").Append(s.RetargetMs.ToString("F1")).Append(" ms");
                if (!double.IsNaN(s.DataAge)) sb.Append("   data age ").Append((s.DataAge * 1000).ToString("F0")).Append(" ms");
                if (!double.IsNaN(s.HarnessPct)) sb.Append("   harness ").Append(s.HarnessPct.ToString("F0")).Append("%");
                sb.Append('\n');
            }
            sb.Append("trackers ").Append(pico.Mode.ToString().ToLowerInvariant()).Append(" (").Append(pico.Status).Append(")   view ")
              .Append(rigView.Mode.ToString().ToLowerInvariant()).Append('\n');
            sb.Append("menu: left menu button");
            return sb.ToString();
        }

        string NoticeText()
        {
            if (!link.Connected)
                return link.Server == null
                    ? "Searching for the simulation server on this network...\nPC: python scripts/sim_server.py --source udp\n(allow UDP 9870 / TCP 9871 in the PC firewall)"
                    : "No answer from " + link.ServerHost + ":" + link.Port + "\nIs sim_server.py running with --source udp?";
            SimState s = states.Latest;
            if (s == null) return "Connected. Waiting for state...";
            if (!robot.Ready && !loadingRobot && cache.LastError != null)
                return "Robot download failed (allow TCP " + (sceneMsg != null ? sceneMsg.AssetPort : 9871) + " in the PC firewall)\n"
                       + cache.LastError + "\nMenu > Server: search again";
            if (!robot.Ready || buildingScene != null) return "Loading " + (robot.Ready ? "scene" : "robot") + "...";
            if (s.Status == "calibrating")
                return "Stand still, arms relaxed:  calibrating " + (int)(s.Progress * 100) + "%" + (string.IsNullOrEmpty(s.Message) ? "" : "\n" + s.Message);
            return string.IsNullOrEmpty(s.Message) ? null : s.Message;
        }

        void BuildMenu()
        {
            hud.Items.Add(new HudItem
            {
                Label = () => "Scene:  < " + (sceneMenuName ?? "?") + " >   (trigger: load)",
                Change = d =>
                {
                    if (sceneNames.Count == 0) return;
                    int i = sceneNames.IndexOf(sceneMenuName);
                    sceneMenuName = sceneNames[((i < 0 ? 0 : i) + d + sceneNames.Count) % sceneNames.Count];
                },
                Activate = () => { if (sceneMenuName != null) link.Send(Protocol.Cmd("load_scene", "scene", sceneMenuName)); },
            });
            hud.Items.Add(new HudItem
            {
                Label = () => "Physics:  < " + (states.Latest != null ? states.Latest.Mode : "?") + " >",
                Change = d => link.Send(Protocol.Cmd("set_mode", "mode", states.Latest != null && states.Latest.Mode == "harness" ? "kinematic" : "harness")),
            });
            hud.Items.Add(new HudItem
            {
                Label = () => "View:  < " + rigView.Mode.ToString().ToLowerInvariant() + " >   (trigger: recenter)",
                Change = d =>
                {
                    rigView.Mode = rigView.Mode == ViewMode.Embodied ? ViewMode.Behind : ViewMode.Embodied;
                    rigView.Recenter();
                    settings.View = rigView.Mode;
                    settings.Save();
                },
                Activate = () => rigView.Recenter(),
            });
            hud.Items.Add(new HudItem
            {
                Label = () => "Robot look:  < " + (settings.GhostRobot ? "ghost" : "solid") + " >",
                Change = d => { settings.GhostRobot = !settings.GhostRobot; settings.Save(); },
            });
            hud.Items.Add(new HudItem
            {
                Label = () => "Trackers:  < " + (pico.Mode == TrackerMode.Objects ? "objects x" + pico.ObjectCount : pico.Mode.ToString().ToLowerInvariant()) + " >",
                Change = d =>
                {
                    TrackerMode m = (TrackerMode)(((int)pico.Mode + d + 3) % 3);
                    pico.SetMode(m);
                    settings.Trackers = m;
                    settings.Save();
                    hud.Toast("Trackers: " + m + " (recalibrate the robot)");
                },
            });
            hud.Items.Add(new HudItem
            {
                Label = () => "Object trackers:  < " + pico.ObjectCount + " >",
                Change = d =>
                {
                    pico.ObjectCount = Mathf.Clamp(pico.ObjectCount + d, 1, 3);
                    settings.ObjectTrackers = pico.ObjectCount;
                    settings.Save();
                    if (pico.Mode == TrackerMode.Objects) pico.SetMode(TrackerMode.Objects);
                },
            });
            hud.Items.Add(new HudItem { Label = () => "Motion Tracker calibration (PICO app)", Activate = () => pico.Calibrate() });
            hud.Items.Add(new HudItem
            {
                Label = () => states.Latest != null && states.Latest.Recording ? "Stop recording" : "Start recording",
                Activate = () => link.Send(Protocol.Cmd("record", "on", !(states.Latest != null && states.Latest.Recording))),
            });
            hud.Items.Add(new HudItem { Label = () => "Recalibrate (stand still)", Activate = () => link.Send(Protocol.Cmd("recalibrate")) });
            hud.Items.Add(new HudItem { Label = () => "Reset to spawn", Activate = () => link.Send(Protocol.Cmd("reset")) });
            hud.Items.Add(new HudItem
            {
                Label = () => "Server:  " + (link.ServerHost ?? "auto") + "   (trigger: search again)",
                Activate = () =>
                {
                    link.SetServer(settings.Server);
                    robotHost = null;
                    sceneMsg = null;
                    cache.LastError = null;
                    hud.Toast("Searching for the server...");
                },
            });
        }
    }
}
