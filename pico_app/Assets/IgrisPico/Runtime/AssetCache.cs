// HTTP downloads from the server's asset port (robot meshes, scene meshes), cached per URL, and mesh
// decoding on a worker thread.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using IgrisPico.Core;
using UnityEngine;
using UnityEngine.Networking;

namespace IgrisPico
{
    public sealed class AssetCache
    {
        readonly Dictionary<string, byte[]> bytes = new Dictionary<string, byte[]>();
        readonly Dictionary<string, Mesh> meshes = new Dictionary<string, Mesh>();
        public string LastError;

        public void Clear()
        {
            bytes.Clear();
            foreach (Mesh m in meshes.Values) UnityEngine.Object.Destroy(m);
            meshes.Clear();
        }

        public IEnumerator Get(string url, Action<byte[]> done)
        {
            byte[] cached;
            if (bytes.TryGetValue(url, out cached))
            {
                done(cached);
                yield break;
            }
            using (UnityWebRequest req = UnityWebRequest.Get(url))
            {
                req.timeout = 20;
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                {
                    LastError = url + ": " + req.error;
                    Debug.LogWarning("[IGRIS] download failed " + LastError);
                    done(null);
                    yield break;
                }
                byte[] data = req.downloadHandler.data;
                bytes[url] = data;
                done(data);
            }
        }

        /// <summary>A Unity mesh for an OBJ / STL file in MuJoCo axes (scaled), or null on failure.</summary>
        public IEnumerator GetMesh(string url, V3 scale, Action<Mesh> done)
        {
            string key = url + "|" + scale;
            Mesh cached;
            if (meshes.TryGetValue(key, out cached))
            {
                done(cached);
                yield break;
            }
            byte[] data = null;
            yield return Get(url, b => data = b);
            if (data == null)
            {
                done(null);
                yield break;
            }
            Task<UnityMeshData> task = Task.Run(() => MeshIO.ToUnity(MeshIO.Load(url, data), scale));
            while (!task.IsCompleted) yield return null;
            if (task.IsFaulted)
            {
                LastError = url + ": " + task.Exception.GetBaseException().Message;
                Debug.LogWarning("[IGRIS] mesh failed " + LastError);
                done(null);
                yield break;
            }
            Mesh mesh = Visuals.BuildMesh(task.Result, System.IO.Path.GetFileNameWithoutExtension(url));
            meshes[key] = mesh;
            done(mesh);
        }
    }
}
