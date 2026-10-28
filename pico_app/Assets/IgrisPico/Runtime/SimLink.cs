// UDP link to the PC simulation server (docs/sim_protocol.md). A background thread receives datagrams;
// the main thread sends and drains the inbox. Until a server is known, "hello" is broadcast on the LAN
// once a second and the first server that answers is used.
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace IgrisPico
{
    /// <summary>Monotonic seconds, usable from any thread.</summary>
    public static class AppClock
    {
        static readonly Stopwatch watch = Stopwatch.StartNew();
        static readonly long epochNs = (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).Ticks * 100;
        public static double Now { get { return watch.Elapsed.TotalSeconds; } }
        /// <summary>Monotonic nanoseconds since the Unix epoch (tracking t_ns).</summary>
        public static long Ns { get { return epochNs + watch.Elapsed.Ticks * 100; } }   // TimeSpan ticks are 100 ns
    }

    public sealed class SimLink : IDisposable
    {
        public struct Packet
        {
            public string Text;
            public IPEndPoint From;
            public double Time;
        }

        public readonly int Port;
        public IPEndPoint Server { get; private set; }
        public double LastReceive = -1;
        public int Sent, Received, Errors;

        readonly UdpClient udp;
        readonly Thread thread;
        readonly ConcurrentQueue<Packet> inbox = new ConcurrentQueue<Packet>();
        volatile bool running = true;
        double lastDiscover = -10;

        public SimLink(int port)
        {
            Port = port;
            udp = new UdpClient(0) { EnableBroadcast = true };
            udp.Client.ReceiveBufferSize = 1 << 20;
            thread = new Thread(ReceiveLoop) { IsBackground = true, Name = "SimLink" };
            thread.Start();
        }

        public string ServerHost { get { return Server == null ? null : Server.Address.ToString(); } }
        public bool Connected { get { return Server != null && LastReceive >= 0 && AppClock.Now - LastReceive < 2.0; } }

        void ReceiveLoop()
        {
            while (running)
            {
                try
                {
                    var from = new IPEndPoint(IPAddress.Any, 0);
                    byte[] data = udp.Receive(ref from);
                    inbox.Enqueue(new Packet { Text = Encoding.UTF8.GetString(data), From = from, Time = AppClock.Now });
                }
                catch (ObjectDisposedException) { return; }
                catch (SocketException)
                {
                    if (!running) return;
                    Interlocked.Increment(ref Errors);   // e.g. ICMP "port unreachable" while the server is down
                }
            }
        }

        public bool TryReceive(out Packet p) { return inbox.TryDequeue(out p); }

        /// <summary>Use host as the server (null / empty: discover again).</summary>
        public bool SetServer(string host)
        {
            IPAddress ip;
            if (string.IsNullOrEmpty(host) || !IPAddress.TryParse(host, out ip))
            {
                Server = null;
                LastReceive = -1;
                return string.IsNullOrEmpty(host);
            }
            Server = new IPEndPoint(ip, Port);
            LastReceive = -1;
            return true;
        }

        /// <summary>Accepts a datagram's sender as the server; false if it is another server than the current one.</summary>
        public bool Accept(Packet p)
        {
            if (Connected && !p.From.Address.Equals(Server.Address)) return false;
            if (Server == null || !Server.Address.Equals(p.From.Address)) Server = new IPEndPoint(p.From.Address, Port);
            LastReceive = p.Time;
            Received++;
            return true;
        }

        public void Send(string json)
        {
            if (Server != null) SendTo(json, Server);
        }

        /// <summary>While not connected: "hello" to the configured server and as a LAN broadcast, once a second.</summary>
        public void Discover(string hello)
        {
            if (Connected || AppClock.Now - lastDiscover < 1.0) return;
            lastDiscover = AppClock.Now;
            if (Server != null) SendTo(hello, Server);
            SendTo(hello, new IPEndPoint(IPAddress.Broadcast, Port));
        }

        void SendTo(string json, IPEndPoint ep)
        {
            byte[] b = Encoding.UTF8.GetBytes(json);
            try
            {
                udp.Send(b, b.Length, ep);
                Sent++;
            }
            catch (SocketException) { Errors++; }
            catch (ObjectDisposedException) { }
        }

        public void Dispose()
        {
            running = false;
            udp.Close();
        }
    }
}
