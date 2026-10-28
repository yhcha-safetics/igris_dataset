// Places the XR rig (tracking space) in the simulation world from state.rig, so the headset view, the
// thumbstick locomotion and the robot share one world.
//   Embodied  the eyes are kept at the robot's head camera: the slowly varying offset between the human head
//             and the robot head (different body proportions) is low-pass corrected.
//   Behind    the rig is moved back so the user watches the robot from behind (recentered on demand).
using IgrisPico.Core;
using UnityEngine;

namespace IgrisPico
{
    public enum ViewMode { Embodied, Behind }

    public sealed class RigView
    {
        public ViewMode Mode = ViewMode.Embodied;
        public float CorrectionTau = 0.5f;       // s
        public float MaxCorrection = 0.6f;       // m
        public Vector3 BehindOffset = new Vector3(0f, 0.3f, -1.8f);   // in the recentered heading frame

        readonly Transform rig, camera;
        Vector3 correction;
        Quaternion behindHeading = Quaternion.identity;
        bool haveHeading;

        public RigView(Transform rig, Transform camera)
        {
            this.rig = rig;
            this.camera = camera;
        }

        public void Reset()
        {
            correction = Vector3.zero;
            haveHeading = false;
        }

        /// <summary>Faces the behind-view offset along the current head direction.</summary>
        public void Recenter() { haveHeading = false; }

        public void Apply(SimState s, float dt)
        {
            if (s == null) return;
            Vector3 p = Visuals.ToUnity(s.Rig.p);
            Quaternion r = Visuals.ToUnity(s.Rig.q);
            Vector3 headLocal = camera.localPosition;   // camera offset frame = tracking space
            if (Mode == ViewMode.Embodied && s.Status == "running")
            {
                Vector3 eye = p + r * headLocal;
                Vector3 err = Visuals.ToUnity(s.HeadCam.p) - eye;
                correction += (err - correction) * (1f - Mathf.Exp(-dt / CorrectionTau));
                correction = Vector3.ClampMagnitude(correction, MaxCorrection);
            }
            else
            {
                correction = Vector3.MoveTowards(correction, Vector3.zero, dt);
            }
            Vector3 offset = correction;
            if (Mode == ViewMode.Behind)
            {
                if (!haveHeading)
                {
                    Vector3 f = r * camera.localRotation * Vector3.forward;
                    f.y = 0f;
                    behindHeading = f.sqrMagnitude > 1e-6f ? Quaternion.LookRotation(f.normalized, Vector3.up) : Quaternion.identity;
                    haveHeading = true;
                }
                offset += behindHeading * BehindOffset;
            }
            rig.SetPositionAndRotation(p + offset, r);
        }
    }
}
