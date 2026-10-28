// PICO Motion Tracker data (PICO Unity Integration SDK 3.x) for the tracking message, in native pose7.
//   Body     full-body tracking (2 or 3 trackers + headset): 24 joints in BodyTrackerRole order. The SDK
//            returns Unity-convention poses in the tracking space (it flips z / qz / qw internally), so
//            they go through Conv.NativeFromUnity like the headset and controllers.
//   Objects  object-tracking mode: up to 3 trackers by id. The SDK returns native poses (the SDK's own
//            sample converts them with ToVector3 / ToQuat for Unity), which is what the message needs.
// The mode must match the one chosen in the PICO Motion Tracker app.
using System.Collections.Generic;
using IgrisPico.Core;
#if IGRIS_PICO
using Unity.XR.PXR;
#endif

namespace IgrisPico
{
    public enum TrackerMode { None, Body, Objects }

    public sealed class PicoTracking
    {
        public TrackerMode Mode { get; private set; }
        public string Status = "off";
        public int ObjectCount = 3;

        readonly List<long> trackerIds = new List<long>();
        long lastBodyTs = long.MinValue;
#if IGRIS_PICO
        BodyTrackingGetDataInfo bodyInfo = new BodyTrackingGetDataInfo();
        BodyTrackingData bodyData = new BodyTrackingData();
        bool subscribed;
#endif

        public void SetMode(TrackerMode mode)
        {
            Stop();
            Mode = mode;
            Status = mode == TrackerMode.None ? "off" : "starting";
            lastBodyTs = long.MinValue;
#if IGRIS_PICO && UNITY_ANDROID && !UNITY_EDITOR
            if (mode == TrackerMode.Body)
            {
                bool supported = false;
                PXR_MotionTracking.GetBodyTrackingSupported(ref supported);
                if (!supported) { Status = "body tracking not supported"; return; }
                int r = PXR_MotionTracking.StartBodyTracking(BodyJointSet.BODY_JOINT_SET_BODY_FULL_START, new BodyTrackingBoneLength());
                Status = r == 0 ? "started" : "start failed (" + r + ")";
            }
            else if (mode == TrackerMode.Objects)
            {
                if (!subscribed)
                {
                    PXR_MotionTracking.RequestMotionTrackerCompleteAction += OnRequestComplete;
                    PXR_MotionTracking.MotionTrackerConnectionAction += OnConnection;
                    subscribed = true;
                }
                // shows the system panel to connect / calibrate the trackers if needed; ids arrive in OnRequestComplete
                int r = PXR_MotionTracking.CheckMotionTrackerNumber((MotionTrackerNum)UnityEngine.Mathf.Clamp(ObjectCount, 1, 3));
                Status = r == 0 ? "waiting for trackers" : "request failed (" + r + ")";
            }
#endif
        }

        void Stop()
        {
#if IGRIS_PICO && UNITY_ANDROID && !UNITY_EDITOR
            if (Mode == TrackerMode.Body) PXR_MotionTracking.StopBodyTracking();
#endif
            Mode = TrackerMode.None;
        }

        /// <summary>Opens the PICO Motion Tracker app (calibration).</summary>
        public void Calibrate()
        {
#if IGRIS_PICO && UNITY_ANDROID && !UNITY_EDITOR
            PXR_MotionTracking.StartMotionTrackerCalibApp();
#endif
        }

#if IGRIS_PICO
        void OnRequestComplete(RequestMotionTrackerCompleteEventData e)
        {
            lock (trackerIds)
            {
                trackerIds.Clear();
                for (int i = 0; i < e.trackerCount && e.trackerIds != null && i < e.trackerIds.Length; i++) trackerIds.Add(e.trackerIds[i]);
            }
        }

        void OnConnection(long id, int state)
        {
            lock (trackerIds)
            {
                if (state == 1 && !trackerIds.Contains(id)) trackerIds.Add(id);
                else if (state == 0) trackerIds.Remove(id);
            }
        }
#endif

        /// <summary>A new full-body sample into body (24 x 7 native) and ts (24); false if none is new.</summary>
        public bool ReadBody(double[] body, long[] ts)
        {
#if IGRIS_PICO && UNITY_ANDROID && !UNITY_EDITOR
            if (Mode != TrackerMode.Body) return false;
            bool tracking = false;
            var st = new BodyTrackingStatus();
            PXR_MotionTracking.GetBodyTrackingState(ref tracking, ref st);
            if (st.stateCode == BodyTrackingStatusCode.BT_INVALID)
            {
                Status = st.message == BodyTrackingMessage.BT_MESSAGE_TRACKER_NOT_CALIBRATED ? "trackers not calibrated" : "no body data (" + st.message + ")";
                return false;
            }
            if (PXR_MotionTracking.GetBodyTrackingData(ref bodyInfo, ref bodyData) != 0 || bodyData.roleDatas == null) return false;
            Status = st.stateCode == BodyTrackingStatusCode.BT_VALID ? "tracking" : "limited";
            long t0 = bodyData.roleDatas[0].localPose.TimeStamp;
            if (t0 == lastBodyTs) return false;   // PICO updates the body slower than the display
            lastBodyTs = t0;
            for (int j = 0; j < TrackingFrame.BodyJoints; j++)
            {
                BodyTrackerTransPose p = bodyData.roleDatas[j].localPose;
                Conv.NativeFromUnity(p.PosX, p.PosY, p.PosZ, p.RotQx, p.RotQy, p.RotQz, p.RotQw, body, 7 * j);
                ts[j] = p.TimeStamp;
            }
            return true;
#else
            return false;
#endif
        }

        /// <summary>Poses (native pose7) of the connected object trackers.</summary>
        public void ReadObjects(List<string> serials, List<double[]> poses)
        {
            serials.Clear();
            poses.Clear();
#if IGRIS_PICO && UNITY_ANDROID && !UNITY_EDITOR
            if (Mode != TrackerMode.Objects) return;
            lock (trackerIds)
            {
                foreach (long id in trackerIds)
                {
                    var loc = new MotionTrackerLocation();
                    bool valid = false;
                    if (PXR_MotionTracking.GetMotionTrackerLocation(id, ref loc, ref valid) != 0 || !valid) continue;
                    serials.Add(id.ToString());
                    poses.Add(new double[] { loc.pose.Position.x, loc.pose.Position.y, loc.pose.Position.z,
                                             loc.pose.Orientation.x, loc.pose.Orientation.y, loc.pose.Orientation.z, loc.pose.Orientation.w });
                }
                Status = trackerIds.Count == 0 ? "waiting for trackers" : poses.Count + "/" + trackerIds.Count + " tracked";
            }
#endif
        }
    }
}
