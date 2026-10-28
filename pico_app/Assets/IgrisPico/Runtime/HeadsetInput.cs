// Headset and controller state in the tracking space (the Camera Offset frame of the rig, Unity axes).
// Without an XR device (Editor play mode) the keyboard and mouse stand in:
//   right mouse drag  look            W A S D  left stick (move)     Q / E  right stick (turn)
//   Space  A   B  B   X  X   Y  Y     G / H  left / right grip      F / J  left / right trigger
//   Tab    left menu                  arrow keys  left stick (menu navigation)
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

namespace IgrisPico
{
    public struct ControllerState
    {
        public bool Valid;
        public Vector3 Pos;
        public Quaternion Rot;
        public bool Primary, Secondary, Menu, StickClick;
        public float Grip, Trigger;
        public Vector2 Stick;
    }

    public sealed class HeadsetInput
    {
        public bool Simulated { get; private set; }
        public bool HeadValid;
        public Vector3 HeadPos;
        public Quaternion HeadRot = Quaternion.identity;
        public ControllerState Left, Right;

        InputDevice head, left, right;
        float simYaw, simPitch;
        static readonly List<InputDevice> devices = new List<InputDevice>();

        public void Poll()
        {
            Simulated = !XRSettings.isDeviceActive;
            if (Simulated)
            {
                PollDesktop();
                return;
            }
            if (!head.isValid) head = Device(XRNode.CenterEye, XRNode.Head);
            if (!left.isValid) left = Device(XRNode.LeftHand, XRNode.LeftHand);
            if (!right.isValid) right = Device(XRNode.RightHand, XRNode.RightHand);

            Vector3 p;
            Quaternion q;
            HeadValid = (head.TryGetFeatureValue(CommonUsages.centerEyePosition, out p) || head.TryGetFeatureValue(CommonUsages.devicePosition, out p))
                        & (head.TryGetFeatureValue(CommonUsages.centerEyeRotation, out q) || head.TryGetFeatureValue(CommonUsages.deviceRotation, out q));
            if (HeadValid)
            {
                HeadPos = p;
                HeadRot = q;
            }
            Read(left, ref Left);
            Read(right, ref Right);
        }

        static InputDevice Device(XRNode node, XRNode fallback)
        {
            InputDevices.GetDevicesAtXRNode(node, devices);
            if (devices.Count == 0 && fallback != node) InputDevices.GetDevicesAtXRNode(fallback, devices);
            return devices.Count > 0 ? devices[0] : default(InputDevice);
        }

        static void Read(InputDevice d, ref ControllerState c)
        {
            if (!d.isValid)
            {
                c = default(ControllerState);
                c.Rot = Quaternion.identity;
                return;
            }
            bool tracked;
            Vector3 p;
            Quaternion q;
            c.Valid = d.TryGetFeatureValue(CommonUsages.devicePosition, out p) & d.TryGetFeatureValue(CommonUsages.deviceRotation, out q)
                      && (!d.TryGetFeatureValue(CommonUsages.isTracked, out tracked) || tracked);
            if (c.Valid)
            {
                c.Pos = p;
                c.Rot = q;
            }
            d.TryGetFeatureValue(CommonUsages.primaryButton, out c.Primary);
            d.TryGetFeatureValue(CommonUsages.secondaryButton, out c.Secondary);
            d.TryGetFeatureValue(CommonUsages.menuButton, out c.Menu);
            d.TryGetFeatureValue(CommonUsages.primary2DAxisClick, out c.StickClick);
            d.TryGetFeatureValue(CommonUsages.grip, out c.Grip);
            d.TryGetFeatureValue(CommonUsages.trigger, out c.Trigger);
            d.TryGetFeatureValue(CommonUsages.primary2DAxis, out c.Stick);
        }

        void PollDesktop()
        {
            if (Input.GetMouseButton(1))
            {
                simYaw += Input.GetAxis("Mouse X") * 3f;
                simPitch = Mathf.Clamp(simPitch - Input.GetAxis("Mouse Y") * 3f, -80f, 80f);
            }
            HeadValid = true;
            HeadPos = new Vector3(0f, 1.6f, 0f);
            HeadRot = Quaternion.Euler(simPitch, simYaw, 0f);
            Quaternion body = Quaternion.Euler(0f, simYaw, 0f);
            Left = SimController(body, -0.2f, Key(KeyCode.X), Key(KeyCode.Y), Key(KeyCode.Tab), Key(KeyCode.G), Key(KeyCode.F),
                                 new Vector2(Axis(KeyCode.D, KeyCode.A) + Axis(KeyCode.RightArrow, KeyCode.LeftArrow),
                                             Axis(KeyCode.W, KeyCode.S) + Axis(KeyCode.UpArrow, KeyCode.DownArrow)));
            Right = SimController(body, 0.2f, Key(KeyCode.Space), Key(KeyCode.B), false, Key(KeyCode.H), Key(KeyCode.J),
                                  new Vector2(Axis(KeyCode.E, KeyCode.Q), 0f));
        }

        ControllerState SimController(Quaternion body, float side, bool primary, bool secondary, bool menu, bool grip, bool trigger, Vector2 stick)
        {
            return new ControllerState
            {
                Valid = true,
                Pos = HeadPos + body * new Vector3(side, -0.45f, 0.35f),
                Rot = body,
                Primary = primary,
                Secondary = secondary,
                Menu = menu,
                Grip = grip ? 1f : 0f,
                Trigger = trigger ? 1f : 0f,
                Stick = Vector2.ClampMagnitude(stick, 1f),
            };
        }

        static bool Key(KeyCode k) { return Input.GetKey(k); }
        static float Axis(KeyCode pos, KeyCode neg) { return (Input.GetKey(pos) ? 1f : 0f) - (Input.GetKey(neg) ? 1f : 0f); }
    }
}
