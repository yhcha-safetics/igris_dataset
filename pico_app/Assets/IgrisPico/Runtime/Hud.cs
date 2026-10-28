// In-headset text: a status panel above the left controller, a notice in front of the eyes (connection,
// calibration, errors, toasts) and the menu (left menu button).
//   menu: left stick up / down select, left / right change, left trigger activate, menu button close
using System;
using System.Collections.Generic;
using UnityEngine;

namespace IgrisPico
{
    public sealed class HudItem
    {
        public Func<string> Label;
        public Action<int> Change;    // -1 / +1 from the stick (optional)
        public Action Activate;       // trigger (optional)
    }

    public sealed class Hud
    {
        public bool MenuOpen { get; private set; }
        public readonly List<HudItem> Items = new List<HudItem>();
        public int Selected;

        readonly Transform head;
        readonly Transform wristPanel, noticePanel, menuPanel, recDot;
        readonly TextMesh status, notice, menu;
        string toast;
        float toastUntil;
        bool prevMenu, prevTrigger;
        int prevStickY, prevStickX;
        float repeatAt;

        public Hud(Transform camera, Transform leftHand)
        {
            head = camera;
            wristPanel = Panel(leftHand, "StatusPanel", new Vector3(0f, 0.08f, 0.02f), new Vector2(0.21f, 0.115f), out status, 0.0016f);
            wristPanel.localRotation = Quaternion.Euler(35f, 0f, 0f);
            menuPanel = Panel(leftHand, "MenuPanel", new Vector3(0f, 0.22f, 0.06f), new Vector2(0.30f, 0.27f), out menu, 0.0019f);
            menuPanel.localRotation = Quaternion.Euler(20f, 0f, 0f);
            menuPanel.gameObject.SetActive(false);
            noticePanel = Panel(null, "Notice", Vector3.zero, new Vector2(0.9f, 0.22f), out notice, 0.0055f, TextAnchor.MiddleCenter);
            recDot = Visuals.Primitive(PrimitiveType.Sphere, null, Visuals.Unlit(new Color(1f, 0.1f, 0.1f, 0.9f), true), "RecordingDot").transform;
            recDot.localScale = Vector3.one * 0.012f;
        }

        static Transform Panel(Transform parent, string name, Vector3 pos, Vector2 size, out TextMesh text, float charSize,
                               TextAnchor anchor = TextAnchor.UpperLeft)
        {
            var t = new GameObject(name).transform;
            t.SetParent(parent, false);
            t.localPosition = pos;
            GameObject bg = Visuals.Primitive(PrimitiveType.Quad, t, Visuals.Unlit(new Color(0.05f, 0.07f, 0.1f, 0.78f), true), "Background");
            bg.transform.localScale = new Vector3(size.x, size.y, 1f);
            bg.transform.localPosition = new Vector3(0f, 0f, 0.001f);
            text = Visuals.Text(t, "Text", charSize, anchor, Color.white);
            text.transform.localPosition = anchor == TextAnchor.MiddleCenter ? Vector3.zero : new Vector3(-size.x / 2 + 0.008f, size.y / 2 - 0.006f, 0f);
            return t;
        }

        public void Toast(string text, float seconds = 2.5f)
        {
            toast = text;
            toastUntil = Time.unscaledTime + seconds;
        }

        /// <summary>Menu navigation from the left controller; returns true while the menu has the left stick.</summary>
        public bool HandleInput(ControllerState left)
        {
            bool menuPressed = left.Menu && !prevMenu;
            prevMenu = left.Menu;
            if (menuPressed) SetMenu(!MenuOpen);
            if (!MenuOpen || Items.Count == 0)
            {
                prevTrigger = left.Trigger > 0.6f;
                return false;
            }
            int sy = left.Stick.y > 0.6f ? 1 : left.Stick.y < -0.6f ? -1 : 0;
            int sx = left.Stick.x > 0.6f ? 1 : left.Stick.x < -0.6f ? -1 : 0;
            bool repeat = sy != 0 && sy == prevStickY && Time.unscaledTime >= repeatAt;
            if (sy != 0 && (sy != prevStickY || repeat))
            {
                Selected = (Selected - sy + Items.Count) % Items.Count;   // stick up = previous item
                repeatAt = Time.unscaledTime + (repeat ? 0.15f : 0.45f);
            }
            prevStickY = sy;
            if (sx != 0 && sx != prevStickX && Items[Selected].Change != null) Items[Selected].Change(sx);
            prevStickX = sx;
            bool trig = left.Trigger > 0.6f;
            if (trig && !prevTrigger && Items[Selected].Activate != null) Items[Selected].Activate();
            prevTrigger = trig;
            return true;
        }

        public void SetMenu(bool open)
        {
            MenuOpen = open;
            menuPanel.gameObject.SetActive(open);
            Selected = Mathf.Clamp(Selected, 0, Mathf.Max(0, Items.Count - 1));
        }

        /// <summary>Per frame: panel texts and the head-locked notice (null / empty: hidden).</summary>
        public void Draw(string statusText, string noticeText, bool recording, float dt)
        {
            status.text = statusText;
            if (MenuOpen)
            {
                var sb = new System.Text.StringBuilder("MENU   (stick: select / change, trigger: run)\n\n");
                for (int i = 0; i < Items.Count; i++)
                    sb.Append(i == Selected ? "> " : "   ").Append(Items[i].Label()).Append('\n');
                menu.text = sb.ToString();
            }
            if (Time.unscaledTime < toastUntil && !string.IsNullOrEmpty(toast))
                noticeText = string.IsNullOrEmpty(noticeText) ? toast : noticeText + "\n" + toast;
            bool show = !string.IsNullOrEmpty(noticeText);
            noticePanel.gameObject.SetActive(show);
            if (show)
            {
                notice.text = noticeText;
                // follows the gaze lazily, 1.6 m ahead, slightly below the eyes
                Vector3 fwd = head.forward;
                fwd.y = Mathf.Clamp(fwd.y, -0.5f, 0.3f);
                Vector3 target = head.position + fwd.normalized * 1.6f + Vector3.down * 0.15f;
                float a = 1f - Mathf.Exp(-dt / 0.25f);
                noticePanel.position = Vector3.Lerp(noticePanel.position, target, noticePanel.position == Vector3.zero ? 1f : a);
                noticePanel.rotation = Quaternion.LookRotation(noticePanel.position - head.position, Vector3.up);
            }
            recDot.gameObject.SetActive(recording);
            if (recording)
            {
                recDot.position = head.position + head.rotation * new Vector3(0.11f, 0.09f, 0.5f);
                recDot.gameObject.SetActive(Mathf.Repeat(Time.unscaledTime, 1f) < 0.7f);
            }
        }
    }
}
