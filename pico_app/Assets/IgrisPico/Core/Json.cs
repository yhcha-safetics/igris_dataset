// Minimal JSON reader / writer for the sim protocol (no Unity dependency, so it is unit-tested with csc).
// Parse() returns Dictionary<string, object>, List<object>, double, string, bool or null.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace IgrisPico.Core
{
    public static class Json
    {
        public static object Parse(string text)
        {
            var p = new Parser(text);
            p.SkipWs();
            object v = p.Value();
            p.SkipWs();
            if (!p.End) throw new FormatException("JSON: trailing characters at " + p.Pos);
            return v;
        }

        // -- accessors that tolerate missing / mistyped fields ------------------------------
        public static Dictionary<string, object> Obj(object o) { return o as Dictionary<string, object>; }
        public static List<object> Arr(object o) { return o as List<object>; }

        public static object Get(Dictionary<string, object> d, string key)
        {
            object v;
            return d != null && d.TryGetValue(key, out v) ? v : null;
        }

        public static double Num(object o, double def = 0.0) { return o is double ? (double)o : def; }
        public static string Str(object o, string def = null) { return o as string ?? def; }
        public static bool Bool(object o, bool def = false) { return o is bool ? (bool)o : def; }

        /// <summary>A list of numbers as double[] (null if o is not a list of numbers).</summary>
        public static double[] Nums(object o)
        {
            var l = o as List<object>;
            if (l == null) return null;
            var a = new double[l.Count];
            for (int i = 0; i < a.Length; i++)
            {
                if (!(l[i] is double)) return null;
                a[i] = (double)l[i];
            }
            return a;
        }

        sealed class Parser
        {
            readonly string s;
            int i;
            public Parser(string text) { s = text ?? ""; }
            public bool End { get { return i >= s.Length; } }
            public int Pos { get { return i; } }

            public void SkipWs()
            {
                while (i < s.Length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
            }

            public object Value()
            {
                if (i >= s.Length) throw new FormatException("JSON: unexpected end");
                char c = s[i];
                if (c == '{') return Object();
                if (c == '[') return Array();
                if (c == '"') return String();
                if (c == 't') { Expect("true"); return true; }
                if (c == 'f') { Expect("false"); return false; }
                if (c == 'n') { Expect("null"); return null; }
                return Number();
            }

            void Expect(string word)
            {
                if (string.CompareOrdinal(s, i, word, 0, word.Length) != 0)
                    throw new FormatException("JSON: unexpected token at " + i);
                i += word.Length;
            }

            Dictionary<string, object> Object()
            {
                var d = new Dictionary<string, object>();
                i++;
                SkipWs();
                if (i < s.Length && s[i] == '}') { i++; return d; }
                while (true)
                {
                    SkipWs();
                    if (i >= s.Length || s[i] != '"') throw new FormatException("JSON: expected key at " + i);
                    string key = String();
                    SkipWs();
                    if (i >= s.Length || s[i] != ':') throw new FormatException("JSON: expected ':' at " + i);
                    i++;
                    SkipWs();
                    d[key] = Value();
                    SkipWs();
                    if (i >= s.Length) throw new FormatException("JSON: unexpected end in object");
                    if (s[i] == ',') { i++; continue; }
                    if (s[i] == '}') { i++; return d; }
                    throw new FormatException("JSON: expected ',' or '}' at " + i);
                }
            }

            List<object> Array()
            {
                var l = new List<object>();
                i++;
                SkipWs();
                if (i < s.Length && s[i] == ']') { i++; return l; }
                while (true)
                {
                    SkipWs();
                    l.Add(Value());
                    SkipWs();
                    if (i >= s.Length) throw new FormatException("JSON: unexpected end in array");
                    if (s[i] == ',') { i++; continue; }
                    if (s[i] == ']') { i++; return l; }
                    throw new FormatException("JSON: expected ',' or ']' at " + i);
                }
            }

            string String()
            {
                i++; // opening quote
                StringBuilder sb = null;
                int start = i;
                while (i < s.Length)
                {
                    char c = s[i];
                    if (c == '"')
                    {
                        string tail = s.Substring(start, i - start);
                        i++;
                        return sb == null ? tail : sb.Append(tail).ToString();
                    }
                    if (c == '\\')
                    {
                        if (sb == null) sb = new StringBuilder();
                        sb.Append(s, start, i - start);
                        if (i + 1 >= s.Length) break;
                        char e = s[i + 1];
                        i += 2;
                        switch (e)
                        {
                            case '"': sb.Append('"'); break;
                            case '\\': sb.Append('\\'); break;
                            case '/': sb.Append('/'); break;
                            case 'b': sb.Append('\b'); break;
                            case 'f': sb.Append('\f'); break;
                            case 'n': sb.Append('\n'); break;
                            case 'r': sb.Append('\r'); break;
                            case 't': sb.Append('\t'); break;
                            case 'u':
                                if (i + 4 > s.Length) throw new FormatException("JSON: bad \\u escape");
                                sb.Append((char)int.Parse(s.Substring(i, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                                i += 4;
                                break;
                            default: throw new FormatException("JSON: bad escape at " + i);
                        }
                        start = i;
                        continue;
                    }
                    i++;
                }
                throw new FormatException("JSON: unterminated string");
            }

            double Number()
            {
                int start = i;
                if (i < s.Length && (s[i] == '-' || s[i] == '+')) i++;
                while (i < s.Length)
                {
                    char c = s[i];
                    if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') i++;
                    else break;
                }
                if (i == start) throw new FormatException("JSON: unexpected character '" + s[i] + "' at " + i);
                return double.Parse(s.Substring(start, i - start), NumberStyles.Float, CultureInfo.InvariantCulture);
            }
        }
    }

    /// <summary>Streaming JSON writer; commas are inserted automatically.</summary>
    public sealed class JsonWriter
    {
        readonly StringBuilder sb = new StringBuilder(2048);
        readonly Stack<bool> first = new Stack<bool>();
        bool afterKey;

        public JsonWriter Clear() { sb.Length = 0; first.Clear(); afterKey = false; return this; }
        public override string ToString() { return sb.ToString(); }

        void Sep()
        {
            if (afterKey) { afterKey = false; return; }
            if (first.Count == 0) return;
            if (first.Peek()) { first.Pop(); first.Push(false); }
            else sb.Append(',');
        }

        public JsonWriter BeginObject() { Sep(); sb.Append('{'); first.Push(true); return this; }
        public JsonWriter EndObject() { sb.Append('}'); first.Pop(); return this; }
        public JsonWriter BeginArray() { Sep(); sb.Append('['); first.Push(true); return this; }
        public JsonWriter EndArray() { sb.Append(']'); first.Pop(); return this; }

        public JsonWriter Key(string k)
        {
            Sep();
            WriteString(k);
            sb.Append(':');
            afterKey = true;
            return this;
        }

        public JsonWriter Value(double v)
        {
            Sep();
            // non-finite numbers are not JSON; the server rejects them anyway
            if (double.IsNaN(v) || double.IsInfinity(v)) sb.Append('0');
            else sb.Append(v.ToString("R", CultureInfo.InvariantCulture));
            return this;
        }

        public JsonWriter Value(float v) { return Value((double)v); }
        public JsonWriter Value(long v) { Sep(); sb.Append(v.ToString(CultureInfo.InvariantCulture)); return this; }
        public JsonWriter Value(int v) { return Value((long)v); }
        public JsonWriter Value(bool v) { Sep(); sb.Append(v ? "true" : "false"); return this; }
        public JsonWriter Null() { Sep(); sb.Append("null"); return this; }

        public JsonWriter Value(string v)
        {
            Sep();
            if (v == null) sb.Append("null");
            else WriteString(v);
            return this;
        }

        public JsonWriter Values(double[] a)
        {
            BeginArray();
            for (int i = 0; i < a.Length; i++) Value(a[i]);
            return EndArray();
        }

        void WriteString(string v)
        {
            sb.Append('"');
            foreach (char c in v)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
        }
    }
}
