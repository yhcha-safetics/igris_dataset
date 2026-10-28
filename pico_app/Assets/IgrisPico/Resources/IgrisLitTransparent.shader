// IgrisPico/Lit with alpha blending (ghost robot, translucent scene parts).
Shader "IgrisPico/LitTransparent"
{
    Properties
    {
        _Color ("Color", Color) = (1, 1, 1, 0.5)
        _MainTex ("Albedo", 2D) = "white" {}
        _Emission ("Emission", Color) = (0, 0, 0, 0)
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "IgnoreProjector" = "True" "RenderType" = "Transparent" }
        LOD 200
        CGPROGRAM
        #pragma surface surf Lambert alpha:fade noforwardadd
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
    FallBack "Legacy Shaders/Transparent/Diffuse"
}
