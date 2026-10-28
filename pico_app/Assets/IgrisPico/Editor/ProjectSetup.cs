// One-time project configuration for PICO 4 Ultra, from the menu (IGRIS > Setup project) or batch mode:
//   Unity.exe -batchmode -quit -projectPath pico_app -executeMethod IgrisPico.Editor.ProjectSetup.Run -logFile -
// Android / IL2CPP / ARM64, PICO XR loader, body tracking permission, LAN HTTP, the Main scene.
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.Management;
#if IGRIS_PICO
using Unity.XR.PXR;
#endif

namespace IgrisPico.Editor
{
    public static class ProjectSetup
    {
        public const string ScenePath = "Assets/IgrisPico/Scenes/Main.unity";
        public const string PackageId = "com.igris.picosim";
        const string PicoLoader = "Unity.XR.PXR.PXR_Loader";
        const string PicoSettingsKey = "Unity.XR.PXR.Settings";

        [MenuItem("IGRIS/1. Setup project")]
        public static void Run()
        {
            if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
                EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);
            ConfigurePlayer();
            ConfigureXR();
            CreateMainScene();
            AssetDatabase.SaveAssets();
            Debug.Log("[IGRIS] project setup done. If 'Active Input Handling' changed, restart the editor before building.");
        }

        static void ConfigurePlayer()
        {
            PlayerSettings.companyName = "IGRIS";
            PlayerSettings.productName = "IGRIS Sim";
            PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.Android, PackageId);
            PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29;
            PlayerSettings.Android.forceInternetPermission = true;
            PlayerSettings.insecureHttpOption = InsecureHttpOption.AlwaysAllowed;   // asset server on the LAN is plain HTTP
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.LandscapeLeft;
            PlayerSettings.runInBackground = true;
            // Input Manager + Input System: the desktop simulation uses UnityEngine.Input, the PICO SDK the Input System
            var ps = AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/ProjectSettings.asset").FirstOrDefault();
            if (ps != null)
            {
                var so = new SerializedObject(ps);
                SerializedProperty p = so.FindProperty("activeInputHandler");
                if (p != null && p.intValue != 2)
                {
                    p.intValue = 2;
                    so.ApplyModifiedProperties();
                }
            }
        }

        static void ConfigureXR()
        {
            XRGeneralSettingsPerBuildTarget perTarget;
            EditorBuildSettings.TryGetConfigObject(XRGeneralSettings.k_SettingsKey, out perTarget);
            if (perTarget == null)
            {
                Directory.CreateDirectory("Assets/XR");
                perTarget = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
                AssetDatabase.CreateAsset(perTarget, "Assets/XR/XRGeneralSettingsPerBuildTarget.asset");
                EditorBuildSettings.AddConfigObject(XRGeneralSettings.k_SettingsKey, perTarget, true);
            }
            if (!perTarget.HasSettingsForBuildTarget(BuildTargetGroup.Android))
                perTarget.CreateDefaultSettingsForBuildTarget(BuildTargetGroup.Android);
            if (!perTarget.HasManagerSettingsForBuildTarget(BuildTargetGroup.Android))
                perTarget.CreateDefaultManagerSettingsForBuildTarget(BuildTargetGroup.Android);
            XRGeneralSettings general = perTarget.SettingsForBuildTarget(BuildTargetGroup.Android);
            general.InitManagerOnStart = true;
            if (!XRPackageMetadataStore.AssignLoader(general.AssignedSettings, PicoLoader, BuildTargetGroup.Android))
                Debug.LogWarning("[IGRIS] could not assign " + PicoLoader + " (is the PICO package installed?)");
            EditorUtility.SetDirty(general);
            EditorUtility.SetDirty(perTarget);
#if IGRIS_PICO
            PXR_Settings pxr;
            EditorBuildSettings.TryGetConfigObject(PicoSettingsKey, out pxr);
            if (pxr == null)
            {
                Directory.CreateDirectory("Assets/XR/Settings");
                pxr = ScriptableObject.CreateInstance<PXR_Settings>();
                AssetDatabase.CreateAsset(pxr, "Assets/XR/Settings/PXR_Settings.asset");
                EditorBuildSettings.AddConfigObject(PicoSettingsKey, pxr, true);
            }
            pxr.stereoRenderingModeAndroid = PXR_Settings.StereoRenderingModeAndroid.Multiview;
            EditorUtility.SetDirty(pxr);
            PXR_ProjectSetting project = PXR_ProjectSetting.GetProjectConfig();
            project.bodyTracking = true;     // manifest feature / permission for PICO Motion Tracker
            PXR_ProjectSetting.SaveAssets();
#endif
        }

        static void CreateMainScene()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var rig = new GameObject("XR Rig");
#if IGRIS_PICO
            PXR_Manager manager = rig.AddComponent<PXR_Manager>();
            manager.bodyTracking = true;
#endif
            var offset = new GameObject("Camera Offset");
            offset.transform.SetParent(rig.transform, false);
            var camGo = new GameObject("Main Camera") { tag = "MainCamera" };
            camGo.transform.SetParent(offset.transform, false);
            var cam = camGo.AddComponent<Camera>();
            camGo.AddComponent<AudioListener>();
            var tpd = camGo.AddComponent<UnityEngine.SpatialTracking.TrackedPoseDriver>();
            tpd.SetPoseSource(UnityEngine.SpatialTracking.TrackedPoseDriver.DeviceType.GenericXRDevice,
                              UnityEngine.SpatialTracking.TrackedPoseDriver.TrackedPose.Center);
            tpd.trackingType = UnityEngine.SpatialTracking.TrackedPoseDriver.TrackingType.RotationAndPosition;
            tpd.updateType = UnityEngine.SpatialTracking.TrackedPoseDriver.UpdateType.UpdateAndBeforeRender;
            var app = new GameObject("IGRIS App").AddComponent<AppMain>();
            app.rig = rig.transform;
            app.xrCamera = cam;
            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
        }
    }
}
