// APK build, from the menu (IGRIS > Build APK) or batch mode:
//   Unity.exe -batchmode -quit -projectPath pico_app -executeMethod IgrisPico.Editor.Builder.BuildAndroid -logFile -
// Output: pico_app/Builds/IgrisSim.apk  (install: adb install -r -g pico_app/Builds/IgrisSim.apk)
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace IgrisPico.Editor
{
    public static class Builder
    {
        public const string Output = "Builds/IgrisSim.apk";

        [MenuItem("IGRIS/2. Build APK")]
        public static void BuildAndroid()
        {
            ProjectSetup.Run();   // idempotent; keeps the generated scene and XR settings current
            Directory.CreateDirectory(Path.GetDirectoryName(Output));
            var options = new BuildPlayerOptions
            {
                scenes = new[] { ProjectSetup.ScenePath },
                locationPathName = Output,
                target = BuildTarget.Android,
                targetGroup = BuildTargetGroup.Android,
                options = BuildOptions.None,
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            BuildSummary s = report.summary;
            Debug.Log("[IGRIS] build " + s.result + ": " + Path.GetFullPath(Output) + " (" + s.totalSize / (1024 * 1024) + " MB, "
                      + s.totalErrors + " errors, " + s.totalWarnings + " warnings)");
            if (Application.isBatchMode) EditorApplication.Exit(s.result == BuildResult.Succeeded ? 0 : 1);
        }
    }
}
