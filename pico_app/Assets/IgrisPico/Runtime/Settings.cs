// User settings, kept in PlayerPrefs. A JSON file in the app's data folder overrides them at start, so the
// server address can be set without the menu:
//   adb push igris_pico.json /sdcard/Android/data/<package>/files/igris_pico.json
//   {"server": "192.168.0.10", "port": 9870, "trackers": "body", "view": "embodied"}
using System;
using System.IO;
using IgrisPico.Core;
using UnityEngine;

namespace IgrisPico
{
    public sealed class Settings
    {
        public string Server = "";               // empty: discover on the LAN
        public int Port = Protocol.DefaultPort;
        public TrackerMode Trackers = TrackerMode.Body;
        public int ObjectTrackers = 3;
        public ViewMode View = ViewMode.Embodied;
        public bool GhostRobot;
        public string Source = "PlayerPrefs";

        public const string FileName = "igris_pico.json";

        public static Settings Load()
        {
            var s = new Settings
            {
                Server = PlayerPrefs.GetString("igris.server", ""),
                Port = PlayerPrefs.GetInt("igris.port", Protocol.DefaultPort),
                Trackers = (TrackerMode)PlayerPrefs.GetInt("igris.trackers", (int)TrackerMode.Body),
                ObjectTrackers = PlayerPrefs.GetInt("igris.object_trackers", 3),
                View = (ViewMode)PlayerPrefs.GetInt("igris.view", (int)ViewMode.Embodied),
                GhostRobot = PlayerPrefs.GetInt("igris.ghost", 0) != 0,
            };
            string path = Path.Combine(Application.persistentDataPath, FileName);
            try
            {
                if (File.Exists(path))
                {
                    var d = Json.Obj(Json.Parse(File.ReadAllText(path)));
                    s.Server = Json.Str(Json.Get(d, "server"), s.Server);
                    s.Port = (int)Json.Num(Json.Get(d, "port"), s.Port);
                    string t = Json.Str(Json.Get(d, "trackers"));
                    if (t != null) s.Trackers = t == "objects" ? TrackerMode.Objects : t == "none" ? TrackerMode.None : TrackerMode.Body;
                    s.ObjectTrackers = (int)Json.Num(Json.Get(d, "object_trackers"), s.ObjectTrackers);
                    string v = Json.Str(Json.Get(d, "view"));
                    if (v != null) s.View = v == "behind" ? ViewMode.Behind : ViewMode.Embodied;
                    s.Source = path;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning("[IGRIS] ignoring " + path + ": " + e.Message);
            }
            return s;
        }

        public void Save()
        {
            PlayerPrefs.SetString("igris.server", Server ?? "");
            PlayerPrefs.SetInt("igris.port", Port);
            PlayerPrefs.SetInt("igris.trackers", (int)Trackers);
            PlayerPrefs.SetInt("igris.object_trackers", ObjectTrackers);
            PlayerPrefs.SetInt("igris.view", (int)View);
            PlayerPrefs.SetInt("igris.ghost", GhostRobot ? 1 : 0);
            PlayerPrefs.Save();
        }
    }
}
