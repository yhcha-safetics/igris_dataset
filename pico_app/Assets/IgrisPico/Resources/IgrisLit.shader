// Lambert-lit colour * texture * vertex colour (scanned scene meshes carry vertex colours).
// In Resources/ so it is always included in the build.
Shader "IgrisPico/Lit"
{
    Properties
    {
        _Color ("Color", Color) = (1, 1, 1, 1)
        _MainTex ("Albedo", 2D) = "white" {}
        _Emission ("Emission", Color) = (0, 0, 0, 0)
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        LOD 200
        CGPROGRAM
        #pragma surface surf Lambert noforwardadd
        #pragma target 3.0
        sampler2D _MainTex;
        fixed4 _Color;
        fixed4 _Emission;
        struct Input
        {
            float2 uv_MainTex;
            float4 color : COLOR;
        };
        void surf (Input IN, inout SurfaceOutput o)
        {
            fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color * IN.color;
            o.Albedo = c.rgb;
            o.Emission = _Emission.rgb;
            o.Alpha = c.a;
        }
        ENDCG
    }
    FallBack "Legacy Shaders/Diffuse"
}
